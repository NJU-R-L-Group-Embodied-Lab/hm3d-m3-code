import cv2
import numpy as np
import os
from habitat import logger
from collections import Sequence
from vlnce_baselines.utils.constant import legend_color_palette
from habitat.core.simulator import Observations
import time
from PIL import Image
import torch

def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image():
    vis_image = np.ones((755, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Goal: "
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color
    
    vis_image = add_class(vis_image, 0, "out of map", legend_color_palette)
    vis_image = add_class(vis_image, 1, "obstacle", legend_color_palette)
    vis_image = add_class(vis_image, 2, "free space", legend_color_palette)
    vis_image = add_class(vis_image, 3, "agent trajecy", legend_color_palette)
    vis_image = add_class(vis_image, 4, "waypoint", legend_color_palette)

    return vis_image


def add_class(vis_image, id, name, color_palette):
    text = f"{id}:{name}"
    font_color = (0, 0, 0)
    class_color = list(color_palette[id])
    class_color.reverse() # BGR -> RGB
    fontScale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = 20, 50
    start_x, start_y = 25, 540 # x is horizontal and point to right, y is vertical and point to down
    gap_x, gap_y = 280, 30
    
    x1 = start_x + (id % 4) * gap_x
    y1 = start_y + (id // 4) * gap_y
    x2 = x1 + w
    y2 = y1 + h
    text_x = x2 + 10
    text_y = (y1 + y2) // 2 + 5
    
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), class_color, thickness=cv2.FILLED)
    vis_image = cv2.putText(vis_image, text, (text_x, text_y),
                            font, fontScale, font_color, thickness, cv2.LINE_AA)
    
    return vis_image


def add_text(image: np.ndarray, text: str, position: Sequence):
    textsize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)[0]
    x, y = position
    x += (480 - textsize[0]) // 2
    image = cv2.putText(image, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, color=0, thickness=1, lineType=cv2.LINE_AA)
    
    return image

ACTION_TO_TEXT = ['stop', 'forward', 'turn left', 'turn right']
ACTION_TO_DISPLAY = {
    0: 'STOP',
    1: 'FORWARD',
    2: 'TURN LEFT',
    3: 'TURN RIGHT',
    'stop': 'STOP',
    'forward': 'FORWARD',
    'turn left': 'TURN LEFT',
    'turn right': 'TURN RIGHT',
}

class LaViRAVisualizer:
    def __init__(self, mapping_module, visualize=False, save_dir="./visualizations", width=640, height=480,
                 debug_log_dir=None):
        self.visualize = visualize
        self.save_dir = save_dir
        self.width = width
        self.height = height
        self.current_episode_id = None
        self.current_step = 0
        if self.visualize:
            os.makedirs(self.save_dir, exist_ok=True)
        self.rgb_history = []
        self.mapping_module = mapping_module
        # When set, every per-step combined image also dumps the topdown VLM map
        # plus the raw RGB into <debug_log_dir>/<episode>/step_<N>/ so that the
        # debug_logs structure has paired prompt/response/images + map view.
        self.debug_log_dir = debug_log_dir

    def sync(self, step, ep_id):
        self.current_step = step
        self.current_episode_id = ep_id

    def update_map(self, mapping_module):
        self.mapping_module = mapping_module

    def _save_rgb_with_bbox(self, rgb_image, bbox, target_coords=None):
        """Save RGB image with bounding box annotation as a separate file
        
        Args:
            rgb_image: RGB image to save
            bbox: Bounding box dictionary
            target_coords: Optional (x, y) tuple of target point to visualize
        """
        if not self.visualize:
            return

        try:
            # Convert RGB image to proper format
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                # Make a copy to avoid modifying the original
                annotated_image = rgb_image.copy()
            else:
                annotated_image = np.array(rgb_image)

            # Draw bounding box
            x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            target_description = bbox.get('target', 'target')
            action_decision = bbox.get('action', 'NAVIGATE')

            # Choose color based on action decision
            box_color = (0, 255, 0) if action_decision == 'NAVIGATE' else (0, 0,
                                                                           255)  # Green for navigate, Red for stop

            # Draw rectangle with thicker border for better visibility
            cv2.rectangle(annotated_image, (x, y), (x + width, y + height), box_color, 4)

            # Draw target point if provided
            if target_coords is not None:
                tx, ty = target_coords
                # Draw red dot
                cv2.circle(annotated_image, (int(tx), int(ty)), 8, (0, 0, 255), -1)
                # Draw white border for better visibility
                cv2.circle(annotated_image, (int(tx), int(ty)), 8, (255, 255, 255), 2)

            # Create footer with "Navigate: XXX"
            action_text = "Navigate" if action_decision == 'NAVIGATE' else action_decision.capitalize()
            footer_text = f"{action_text}: {target_description}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Dynamic font scaling
            h, w = annotated_image.shape[:2]
            max_text_width = w - 40 # Padding of 20px on each side
            
            (text_width, text_height), baseline = cv2.getTextSize(footer_text, font, font_scale, thickness)
            
            if text_width > max_text_width:
                font_scale = font_scale * (max_text_width / text_width)
                (text_width, text_height), baseline = cv2.getTextSize(footer_text, font, font_scale, thickness)
            
            footer_height = int(text_height * 2.5) # Dynamic footer height
            
            # Extend image
            new_h = h + footer_height
            
            new_image = np.zeros((new_h, w, 3), dtype=np.uint8) # Black background
            new_image[:h, :, :] = annotated_image
            
            # Center text in footer
            text_x = max(20, (w - text_width) // 2)
            text_y = h + (footer_height + text_height) // 2 - int(text_height * 0.2)
            
            cv2.putText(new_image, footer_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
            
            annotated_image = new_image

            # Save to episode-specific folder with chronological naming
            episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"
            img_folder = os.path.join(self.save_dir, episode_id)

            # Create episode folder if it doesn't exist
            os.makedirs(img_folder, exist_ok=True)

            # Use current step for chronological ordering
            filename = f"bbox_step{self.current_step:04d}.png"
            filepath = os.path.join(img_folder, filename)

            # Convert RGB to BGR for cv2 saving (cv2 expects BGR)
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, annotated_image_bgr)

            # logger.info(f"Saved RGB with bbox annotation: {filepath}")

        except Exception as e:
            logger.info(f"Error saving RGB with bbox: {e}")

    def _save_depth(
            self,
            depth_image: np.ndarray,
            step: int,
            invert: bool = False,
            save_16bit: bool = False,
            use_colormap: bool = True,
            colormap: int = cv2.COLORMAP_JET
    ):
        """
        Save depth:
          - If save_16bit=True: 16-bit single channel PNG (no colormap).
          - Else if use_colormap=True: colored PNG using OpenCV colormap (default Jet).
          - Else: 8-bit grayscale PNG.
        Args:
            depth_image: float32 depth normalized or raw in [0,1] (will be clipped).
            invert: if True, near becomes bright.
            save_16bit: save a 16-bit raw depth (no colormap).
            use_colormap: apply colormap (ignored if save_16bit=True).
            colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).
        Returns:
            np.ndarray saved array (8-bit color/grayscale or 16-bit).
        """
        if not self.visualize:
            return None

        try:
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth, 0.0, 1.0)
            if invert:
                depth = 1.0 - depth

            episode_id = str(self.current_episode_id) if getattr(self, "current_episode_id", None) else "unknown"
            out_dir = os.path.join(self.save_dir, episode_id)
            os.makedirs(out_dir, exist_ok=True)

            if save_16bit:
                depth_u16 = (depth * 65535.0 + 0.5).astype(np.uint16)
                # fname = f"depth16_step{step:04d}.png"
                # path = os.path.join(out_dir, fname)
                # cv2.imwrite(path, depth_u16)
                return depth_u16

            depth_u8 = (depth * 255.0 + 0.5).astype(np.uint8)

            if use_colormap:
                colored = cv2.applyColorMap(depth_u8, colormap)
                # fname = f"depth_color_step{step:04d}.png"
                # path = os.path.join(out_dir, fname)
                # cv2.imwrite(path, colored)
                return colored
            else:
                # fname = f"depth_gray_step{step:04d}.png"
                # path = os.path.join(out_dir, fname)
                # cv2.imwrite(path, depth_u8)
                return depth_u8
        except Exception as e:
            logger.info(f"Error saving depth: {e}")
            return None

    def _create_combined_image(self, rgb_image, metadata, step, visited_targets, target_coords=None, reasoning_info=None, todo_list=None, iplanner_traj=None, la_output=None):
        """Create a combined image with RGB on left, VLM map on right, instruction text, and reasoning info

        Args:
            rgb_image: RGB observation image
            metadata: Metadata dict containing episode info
            step: Current step number
            target_coords: Tuple of (target_map_x, target_map_y) or None
            reasoning_info: Dict containing reasoning and progress_analysis from LLM
            todo_list: String containing the Markdown checklist
            iplanner_traj: List of (x, y) tuples in world coordinates
            la_output: Dict containing output from LA model (action, reasoning, etc.)
        """
        # Use la_output as reasoning_info if provided
        if la_output:
            reasoning_info = la_output

        # Create goal tensor for visualization if target coordinates are available
        goal_tensor = None
        if target_coords is not None and target_coords[0] is not None and target_coords[1] is not None:
            goal_tensor = torch.tensor([target_coords[0], target_coords[1]], dtype=torch.float32)

        # Generate VLM map
        vlm_map = self.mapping_module.create_vlm_map_from_state(
            self.current_episode_id, 0, goal_tensor,
            output_size=(1024, 1024), visited_targets=visited_targets, display_last=True,
            iplanner_traj=iplanner_traj
        ).copy()

        debug_map_path = os.path.join(self.save_dir, str(self.current_episode_id), f'debug_vlm_map_step{step:04d}.png')
        # os.makedirs(os.path.dirname(debug_map_path), exist_ok=True)
        # cv2.imwrite(debug_map_path, vlm_map)

        # When DEBUG_LOGGING is on, dump a per-step topdown map next to the
        # LA/VA prompt/response under the structured debug log dir.
        if self.debug_log_dir is not None and self.current_episode_id is not None:
            try:
                _step_dir = os.path.join(self.debug_log_dir, str(self.current_episode_id), f'step_{step}')
                os.makedirs(_step_dir, exist_ok=True)
                # vlm_map is BGR from create_vlm_map_from_state; cv2.imwrite expects BGR.
                cv2.imwrite(os.path.join(_step_dir, 'topdown_map.png'), vlm_map)
                # Also save the raw RGB observation alongside for parity with the map.
                if isinstance(rgb_image, np.ndarray):
                    rgb_to_save = rgb_image
                    if rgb_to_save.dtype != np.uint8:
                        rgb_to_save = (rgb_to_save * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(_step_dir, 'rgb_obs.png'),
                                cv2.cvtColor(rgb_to_save, cv2.COLOR_RGB2BGR))
            except Exception as _e:
                logger.warning(f'failed to save step debug map/rgb: {_e}')

        # Convert RGB image to proper format if needed
        if isinstance(rgb_image, np.ndarray):
            rgb_display = rgb_image.copy()
        else:
            rgb_display = np.array(rgb_image)

        # Ensure RGB is in correct format (H, W, 3)
        if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
            pass  # Already correct format
        else:
            logger.info(f"Warning: Unexpected RGB shape: {rgb_display.shape}")

        # Convert VLM map from BGR to RGB for consistent display
        if len(vlm_map.shape) == 3:
            vlm_map_display = cv2.cvtColor(vlm_map, cv2.COLOR_BGR2RGB)
        else:
            vlm_map_display = vlm_map.copy()

        # Get target height from RGB image
        target_height = rgb_display.shape[0]

        # Resize VLM map to match RGB height while preserving aspect ratio
        vlm_old_h, vlm_old_w = vlm_map_display.shape[:2]
        if vlm_old_h == 0 or vlm_old_w == 0:
            vlm_target_width = target_height
        else:
            vlm_target_width = int(float(target_height) / vlm_old_h * vlm_old_w)

        # Resize VLM map
        vlm_map_resized = cv2.resize(
            vlm_map_display,
            (vlm_target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )

        # Pad images to same height with white background
        def pad_to_height(img, target_height):
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            if img.shape[0] < target_height:
                pad_height = target_height - img.shape[0]
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                return np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)
            return img

        rgb_padded = pad_to_height(rgb_display, target_height)
        vlm_map_padded = pad_to_height(vlm_map_resized, target_height)

        # Combine horizontally: RGB on left, VLM map on right
        frame = np.concatenate((rgb_padded, vlm_map_padded), axis=1)

        # Prepare metadata for text display
        instruction = metadata.get('instruction', 'No instruction')
        destination = metadata.get('destination', 'unknown')
        action = metadata.get('action', 'unknown')

        # Get LA Action from reasoning_info
        la_action_text = ""
        if reasoning_info:
             action = reasoning_info.get('action', reasoning_info.get('decision', ''))
             direction = reasoning_info.get('direction', '')
             stop_signal = reasoning_info.get('stop_signal', False)
             
             if action == 'NAVIGATE' and direction:
                 la_action_text = f"navigate to {direction}"
             elif action == 'BACKTRACK':
                 waypoint = reasoning_info.get('waypoint', '')
                 la_action_text = f"backtrack to {waypoint}"
             else:
                 la_action_text = action
                 
             if stop_signal:
                 la_action_text += " (STOP)"
        
        if la_action_text:
             instruction_display = f"Instruction: {instruction}  [LA: {la_action_text}]"
        else:
             instruction_display = f"Instruction: {instruction}"

        action_display = 'UNKNOWN'
        try:
            if isinstance(action, dict):
                action = action.get('action', '')

            if isinstance(action, str) and action.isdigit():
                action = int(action)
            elif isinstance(action, str):
                pass

            # Try to get display action text
            if isinstance(action, int) and 0 <= action < len(ACTION_TO_TEXT):
                action_display = ACTION_TO_DISPLAY.get(action, ACTION_TO_TEXT[action].upper())
            elif isinstance(action, str):
                action_display = ACTION_TO_DISPLAY.get(action, action.upper())
            else:
                action_display = 'UNKNOWN'
        except Exception as e:
            logger.info(f"Error parsing action: {e}")
            action_display = 'UNKNOWN'

        # Create instruction area - increased height to accommodate reasoning info and todo list
        instruction_height = 500  # Increased to accommodate TODO list
        white_bg = np.ones((instruction_height, frame.shape[1], 3), dtype=np.uint8) * 255

        # Text rendering setup
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 25

        # Function to wrap long text
        def wrap_text(text, max_width, font, font_scale, thickness):
            if not text:
                return [text]

            words = text.split(' ')
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]

                if text_size[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        lines.append(word)

            if current_line:
                lines.append(current_line)

            return lines if lines else [text]

        # Calculate max width for text
        max_text_width = frame.shape[1] - 30

        y_pos = 20
        base_text_lines = [
            f"Step: {step} | Episode: {metadata.get('episode_id', 'N/A')} | ACTION: {action_display}",
            f"Destination: {destination}",
            "",
            instruction_display
        ]

        # Draw text lines
        for line in base_text_lines:
            if line.strip():
                color = (0, 0, 0)
                current_font_scale = font_scale
                current_thickness = thickness

                if "Step:" in line and "ACTION:" in line:
                    color = (0, 0, 255)
                    current_font_scale = 0.7
                    current_thickness = 2
                elif "Destination:" in line:
                    color = (0, 165, 255)
                elif "Instruction:" in line:
                    color = (255, 0, 0)
                    # Wrap instruction text
                    if len(line) > 20:
                        instruction_text = line.replace("Instruction: ", "")
                        wrapped_instruction = wrap_text(instruction_text, max_text_width - 120, font,
                                                        current_font_scale, current_thickness)

                        # Draw "Instruction: " first
                        cv2.putText(
                            white_bg,
                            "Instruction: ",
                            (15, y_pos),
                            font,
                            current_font_scale,
                            color,
                            current_thickness,
                            lineType=cv2.LINE_AA
                        )
                        y_pos += line_height

                        # Draw wrapped lines with indentation
                        for wrapped_line in wrapped_instruction:
                            cv2.putText(
                                white_bg,
                                wrapped_line,
                                (30, y_pos),
                                font,
                                current_font_scale,
                                color,
                                current_thickness,
                                lineType=cv2.LINE_AA
                            )
                            y_pos += line_height
                        continue

                # Regular text drawing
                cv2.putText(
                    white_bg,
                    line,
                    (15, y_pos),
                    font,
                    current_font_scale,
                    color,
                    current_thickness,
                    lineType=cv2.LINE_AA
                )
            y_pos += line_height

        # Add reasoning and progress analysis information if available
        if reasoning_info:
            y_pos += 5  # Add some spacing

            # 1. LA Action / Decision (REMOVED as requested, moved to Instruction line)
            
            # Draw progress analysis (REMOVED as requested)
            
            # Draw reasoning (REMOVED as requested)
            pass

        # Draw TODO List if available
        if todo_list:
            y_pos += 5  # Add some spacing
            cv2.putText(
                white_bg,
                "TODO List: ",
                (15, y_pos),
                font,
                0.5,  # Slightly smaller font
                (255, 140, 0),  # Dark Orange
                1,
                lineType=cv2.LINE_AA
            )
            y_pos += line_height

            # Split TODO list into lines
            todo_lines = todo_list.split('\n')
            for line in todo_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for completion status
                is_completed = '- [x]' in line or '- [X]' in line
                color = (100, 100, 100) if is_completed else (0, 0, 0) # Gray if done, Black if todo
                
                # Wrap long lines
                wrapped_todo = wrap_text(line, max_text_width - 60, font, 0.5, 1)
                for wrapped_line in wrapped_todo:
                    if y_pos < instruction_height - 15:
                        cv2.putText(
                            white_bg,
                            wrapped_line,
                            (30, y_pos),
                            font,
                            0.5,
                            color,
                            1,
                            lineType=cv2.LINE_AA
                        )
                        y_pos += 18

        # Create header for the two panels
        header_height = 25
        header_bg = np.ones((header_height, frame.shape[1], 3), dtype=np.uint8) * 240

        rgb_width = rgb_padded.shape[1]
        vlm_width = vlm_map_padded.shape[1]

        headers = ["RGB View", "VLM Navigation Map"]
        header_colors = [(0, 100, 0), (0, 0, 139)]  # Dark green, Dark red

        # Position headers in center of each panel
        x_positions = [
            rgb_width // 2,  # RGB center
            rgb_width + vlm_width // 2  # VLM map center
        ]

        for i, (header, color, x_pos) in enumerate(zip(headers, header_colors, x_positions)):
            text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_centered = max(0, x_pos - text_size[0] // 2)

            cv2.putText(
                header_bg,
                header,
                (x_centered, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA
            )

        # Combine all components: header + main frame + instruction area
        final_frame = np.concatenate([header_bg, frame, white_bg], axis=0)

        return Image.fromarray(final_frame.astype(np.uint8))

    def _save_rgb_frame(self, obs: Observations, step: int, visited_targets, episode_id: str = None, target_coords: tuple = None, todo_list: str = None, iplanner_traj: list = None, la_output: dict = None):
        """Save RGB frame with metadata, instruction text and top-down map"""
        if not self.visualize:
            return

        rgb_image = obs['rgb']
        episode_id = episode_id or self.current_episode_id
        episode_id = str(episode_id) if not isinstance(episode_id, str) else episode_id

        metadata = {
            'episode_id': episode_id,
            'step': step,
            'timestamp': time.time(),
            'instruction': getattr(self, 'instruction', ''),
            'pose': obs.get('sensor_pose', None),
            'destination': getattr(self, 'destination', 'unknown'),
            'action': getattr(self, '_action', 'unknown')
        }

        # Save the RGB frame as a separate file
        self._save_individual_rgb(rgb_image, episode_id, step)

        # logger.info(target_coords)
        # Create combined image with RGB, top-down map, and value map heatmap
        combined_image = self._create_combined_image(rgb_image, metadata, step, visited_targets, target_coords, todo_list=todo_list, iplanner_traj=iplanner_traj, la_output=la_output)

        img_folder = os.path.join(self.save_dir, episode_id)
        img_path = os.path.join(img_folder, f"combined_step{step:04d}.png")
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        combined_image.save(img_path)

    def _save_individual_rgb(self, rgb_image, episode_id, step):
        """Save individual RGB image as a separate file"""
        pass
        # try:
        #     episode_id = str(episode_id) if episode_id else "unknown"

        #     # Create episode-specific RGB folder
        #     rgb_folder = os.path.join(self.save_dir, episode_id, "rgb_only")
        #     os.makedirs(rgb_folder, exist_ok=True)

        #     # Convert RGB image to proper format
        #     if isinstance(rgb_image, np.ndarray):
        #         rgb_display = rgb_image.copy()
        #     else:
        #         rgb_display = np.array(rgb_image)

        #     # Save RGB image
        #     filename = f"rgb_step{step:04d}.png"
        #     filepath = os.path.join(rgb_folder, filename)

        #     # Convert RGB to BGR for cv2 saving (cv2 expects BGR)
        #     if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
        #         rgb_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)
        #         cv2.imwrite(filepath, rgb_bgr)
        #     else:
        #         # Fallback: save as PIL Image
        #         if isinstance(rgb_image, np.ndarray):
        #             Image.fromarray(rgb_image).save(filepath)
        #         else:
        #             rgb_image.save(filepath)

        # except Exception as e:
        #     logger.info(f"Error saving individual RGB: {e}")

    def _save_waypoint_panorama_rgb(self, panorama_frames, waypoint_id, step):
        """Save the 4 directional RGB images for a waypoint"""
        pass
        # if not self.visualize:
        #     return

        # try:
        #     episode_id = str(self.current_episode_id) if self.current_episode_id else "unknown"

        #     # Create waypoint-specific folder
        #     waypoint_folder = os.path.join(self.save_dir, episode_id, "waypoints",
        #                                    f"waypoint_{waypoint_id:02d}_step{step:04d}")
        #     os.makedirs(waypoint_folder, exist_ok=True)

        #     # Direction names for the 4 cardinal directions
        #     direction_names = ['forward', 'left', 'behind', 'right']

        #     for i, frame in enumerate(panorama_frames):
        #         if i >= 4:  # Only save first 4 frames (forward, left, behind, right)
        #             break

        #         rgb_image = frame['rgb']
        #         direction_name = direction_names[i]
        #         angle = frame.get('angle', i * 90)

        #         # Convert RGB image to proper format
        #         if isinstance(rgb_image, np.ndarray):
        #             rgb_display = rgb_image.copy()
        #         else:
        #             rgb_display = np.array(rgb_image)

        #         # Save RGB image with direction name
        #         filename = f"{direction_name}_{angle:03d}deg.png"
        #         filepath = os.path.join(waypoint_folder, filename)

        #         # Convert RGB to BGR for cv2 saving
        #         if len(rgb_display.shape) == 3 and rgb_display.shape[2] == 3:
        #             rgb_bgr = cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR)
        #             cv2.imwrite(filepath, rgb_bgr)
        #         else:
        #             # Fallback: save as PIL Image
        #             if isinstance(rgb_image, np.ndarray):
        #                 Image.fromarray(rgb_image).save(filepath)
        #             else:
        #                 rgb_image.save(filepath)

        #     logger.info(f"Saved waypoint {waypoint_id} panorama RGB images to: {waypoint_folder}")

        # except Exception as e:
        #     logger.info(f"Error saving waypoint panorama RGB: {e}")
