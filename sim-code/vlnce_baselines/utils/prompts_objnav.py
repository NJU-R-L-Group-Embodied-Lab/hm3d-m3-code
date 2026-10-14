
LA_PROMPT_BACKTRACK = """Based on the navigation history, current 4-directional views, and the TODO list,
decide the next action to find the target object. If appropriate, suggest incremental updates to the TODO list.

Current TODO List:
{todo_list}

{feedback}

Task:
1. FIRST, write `reasoning_todo`: review ALL TODO items (not just the first one) and
   decide what to update. Items have only two statuses: "pending" and "completed".
   Sub-goals may be completed out of order or implicitly as you navigate.
   - If yes: in `todo_updates`, mark it `completed` with a concrete `result` describing
     the observation that confirms completion (e.g., "Saw kitchen through archway at
     Waypoint 3").
   - IMPORTANT: If you mark a LATER item `completed`, re-examine EARLIER items —
     they may have been done implicitly while you traversed.
   - If an item no longer matches what you see, use `op:rewrite` to refine it.
   - If a new sub-goal emerges, `op:add` it; omit `index` to append at the end, or
     include `index` to inject at that position (later items shift down).
   - If an item turns out to be irrelevant / unreachable, `op:remove` it.
   - `completed` WITHOUT a concrete `result` will be rolled back.

2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO
   plus what you observe.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "map_understanding": "<if a team top-down map is provided: where YOU are by YOUR color in the legend, teammate pose/heading by their colors, FOUND from the text legend if any, N/S/E/W direction of next place (up=North); else empty string>",
    "progress_analysis": "<brief assessment of what you've observed and done so far>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [  // include any needed updates; use [] if nothing changes
        {{"index": <int>, "status": "completed", "result": "<REQUIRED observation>"}},
        {{"op": "rewrite", "index": <int>, "content": "<refined description>"}},
        {{"op": "add", "content": "<new sub-goal>", "status": "pending", "index": <int (optional; omit to append at end)>}},
        {{"op": "remove", "index": <int>}}
    ],
    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": "up"|"down"|False
}}

Guidelines:
- Do NOT try to open the doors.
- If you are going up/down stairs (or steps), indicate "stair": "up" or "down", otherwise False.
- Set stop=true whenever you have visual confirmation of the target within ~1m. The TODO
  list does NOT gate stopping — you may stop regardless of how many pending items remain.
- Focus on finding the target object specified in the task.
- When a team top-down map is present, fill map_understanding FIRST (find YOUR color in the legend, teammates by color, FOUND from the text legend if any, N/S/E/W where you will go; up=North). Colored X is only a short-term waypoint, not a FOUND object.
{negative_constraints}"""


LA_PROMPT_BACKTRACK_REPLAN = """You are a navigation agent. You have backtracked to a previous waypoint to have a second chance to choose action.

Instruction: "{instruction}"

Navigation History (up to this waypoint):
(Images provided above)

Current 4-directional views at this waypoint:
(Images provided above)

Previous Action: **navigate to {previous_action}** from here.

Previous Trajectory (Path taken from here):
(Images provided above)

Task:
- Review the Previous Trajectory to understand the outcome of the previous choice.
- Analyze the Current 4-directional views to give a second-chance choice.
- The previous choice can be reconsidered if necessary.

Available Actions:
{action_list}

Response format (JSON):
{{
    "reasoning": "<analysis of the previous path and why the new direction is chosen>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": "up"|"down"|False
}}

Guidelines:
- Do NOT try to open the doors.
- If you believe you have reached the target object at this waypoint, set "stop": True.
- If going up/down stairs (or steps), indicate "stair": "up" or "down".
- Focus on finding the target object specified in the task. 
{negative_constraints}"""

LA_PROMPT_TODO_GENERATOR = """You are a navigation agent.
Instruction: "{instruction}"

Create a TODO list to complete this instruction.

Response format (JSON ONLY):
{{
    "todos": [
        {{"content": "<step description>", "status": "pending"}},
        {{"content": "<step description>", "status": "pending"}},
        ...
    ]
}}

Return ONLY the JSON object.
"""

LA_PROMPT_BACKTRACK_OBJ = """Based on the navigation history and current 4-directional views, decide the next action to find the target object.

Review the "Navigation History" provided above to identify available waypoints for backtracking. 
Each waypoint in the history (e.g., "Waypoint 0", "Waypoint 1") is a potential candidate.

Choose one of these actions:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward  
3. navigate to right - turn right and go forward
4. navigate to behind - turn around and go forward
5. backtrack to <waypoint_id> - return to a previous waypoint

Response format (JSON):
{{
    "progress_analysis": "<assessment of search progress (e.g., target spotted, room type identified)>",
    "reasoning": "<explanation of chosen action based on semantic clues>",
    "action": "navigate to forward|left|right|behind" or "backtrack to <waypoint_id>",
    "stair": True/False
}}

Guidelines:
- Focus on finding the target object specified in the task.
- Navigate towards areas that are semantically likely to contain the target (e.g., go to kitchen for a toaster).
- If the target is visible, move towards it.
- Use backtracking if the current area is fully explored or irrelevant.
- If you are going or currently climbing stairs, indicate "stair": True otherwise False"""

LA_PROMPT_BACKTRACK_OLD = """Based on the navigation history and current 4-directional views, decide the next action:

Review the "Navigation History" provided above to identify available waypoints for backtracking. 
Each waypoint in the history (e.g., "Waypoint 0", "Waypoint 1") is a potential candidate.

Choose one of these actions:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward  
3. navigate to right - turn right and go forward
4. navigate to behind - turn around and go forward
5. backtrack to <waypoint_id> - return to a previous waypoint

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen action>",
    "action": "navigate to forward|left|right|behind" or "backtrack to <waypoint_id>",
    "stair": True/False
}}

Guidelines:
- Consider instruction completion progress
- Choose direction that best advances toward goal
- If you are going or currently climbing stairs, indicate "stair": True otherwise False"""

LA_PROMPT_NO_BACKTRACK_VLN = """Based on the navigation history and current 4-directional views, decide the next navigation direction:

Choose the best direction:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward
3. navigate to right - turn right and go forward  
4. navigate to behind - turn around and go forward

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of chosen direction>",
    "action": "{action_desc}",
    "stair": True/False
}}

Guidelines:
- Consider instruction completion progress
- Choose direction that best advances toward goal
- Look for relevant objects and clear paths
- If you are going or currently climbing stairs, indicate "stair": True otherwise False"""

LA_PROMPT_NO_BACKTRACK = """Based on the navigation history, current 4-directional views, and the TODO list,
decide the next navigation direction. If appropriate, suggest incremental updates to the TODO list.

Current TODO List:
{todo_list}

{feedback}

Task:
1. FIRST, write `reasoning_todo` and `todo_updates`. Items have only two statuses:
   "pending" and "completed". Review ALL items (not just the first one); a sub-goal
   may be completed out of order.
   - Mark an item `completed` if you judge it done (with a concrete `result`).
   - If marking a LATER item `completed`, re-check EARLIER items.
   - Use `op:rewrite` to refine; `op:add` to introduce a new sub-goal (optional
     `index` to inject at a position; omit to append); `op:remove` to drop.
   - `completed` without a concrete `result` is rolled back.

2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO
   plus what you observe.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "map_understanding": "<if a team top-down map is provided: where YOU are by YOUR color in the legend, teammate pose/heading by their colors, FOUND from the text legend if any, N/S/E/W direction of next place (up=North); else empty string>",
    "progress_analysis": "<brief assessment of search progress>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [
        {{"index": <int>, "status": "completed", "result": "<REQUIRED observation>"}},
        {{"op": "rewrite", "index": <int>, "content": "<refined description>"}},
        {{"op": "add", "content": "<new task>", "status": "pending", "index": <int (optional; omit to append)>}},
        {{"op": "remove", "index": <int>}}
    ],
    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",
    "action": "{action_desc}",
    "stop": False,
    "stair": "up"|"down"|False
}}

Guidelines:
- Focus on finding the target object specified in the task.
- If you are going up/down stairs (or steps), indicate "stair": "up" or "down", otherwise False
- When a team top-down map is present, fill map_understanding FIRST (find YOUR color in the legend, teammates by color, FOUND from the text legend if any, N/S/E/W where you will go; up=North). Colored X is only a short-term waypoint, not a FOUND object.
{negative_constraints}"""


VA_PROMPT = """Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
- Image size: {width}x{height} pixels
- Current TODO List:
{todo_list}
- Progress Info: {progress_info}

Your task:
1. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
2. Provide the bounding box of the target.

Response format (JSON):
{{
    "reasoning": "<brief explanation of decision>",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- Target description should be specific and clear.
- If deciding to go up or down stairs, the bbox MUST select the ENTRY of the stairs.
- Provide bounding box of next target."""

STOP_CHECK_PROMPT = """You are an intelligent navigation agent.
Task: "{instruction}"

You have decided to STOP, believing you have reached the target.
Now, please double-check your decision based on the current 4-directional views and the previous waypoint's 4-directional views.

Previous Waypoint Views:
{previous_views}

Current Views:
{current_views}

Requirements:
1. Check if the target object is clearly visible in any of the views.
2. Estimate if the distance to the target is less than 1 meter.
3. Compare previous views with current views to confirm you have approached the target.

Decision Rules:
- If target is visible AND distance < 1m: CONFIRM STOP.
- If target is NOT visible OR distance >= 1m: CONTINUE NAVIGATION.

Response format (JSON):
{{
    "analysis": "<analysis of visibility and distance, and comparison with previous views>",
    "decision": "STOP" or "CONTINUE"
}}
"""

# ==== Additional ObjectNav-specific prompts ====


LA_PROMPT_OBJECTNAV = """Task: Find the "{target_object}".

Based on the navigation history and current 4-directional views, decide the next action to find the target.

Review the "Navigation History" provided above to identify available waypoints for backtracking.
Each waypoint in the history (e.g., "Waypoint 0", "Waypoint 1") is a potential candidate.

Analysis:
1. **Target Identification**: Is the "{target_object}" visible in any current view?
2. **Room Type Inference**: Based on objects seen, where are you? Is the target likely here?
3. **Exploration**: If target is not seen, choose a direction that leads to a *new* area or a room likely to contain the target.

Choose one of these actions:
1. navigate to forward - continue straight ahead
2. navigate to left - turn left and go forward
3. navigate to right - turn right and go forward
4. navigate to behind - turn around and go forward
5. backtrack to <waypoint_id> - return to a previous waypoint

Response format (JSON):
{{
    "progress_analysis": "<searching/spotted/exploring>",
    "reasoning": "<explanation of chosen action>",
    "action": "navigate to forward|left|right|behind" or "backtrack to <waypoint_id>",
    "stair": True/False
}}

Guidelines:
- If "{target_object}" is visible, go towards it immediately.
- If not, go towards the most open path or a door leading to a new room.
- Consider backtracking if you have explored the current area sufficiently.
- Use "stair": True if you are using stairs.
"""


# NOTE: The first VA_PROMPT (line 197) uses {todo_list} and matches query_llm's
# call signature. The variant below (with {visited_targets_str}) is kept only as
# VA_PROMPT_VISITED_TARGETS for reference — we do NOT override VA_PROMPT here.
VA_PROMPT_VISITED_TARGETS = """Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
- Image size: {width}x{height} pixels{visited_targets_str}{progress_info}

Your task:
1. Analyze at what stage the current instruction has been completed and what should be done next
2. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
3. Decide if the robot should STOP (if task is completed or very close to final goal)

Response format (JSON):
{{
    "progress": "<assessment of how close to completing the instruction>",
    "reasoning": "<brief explanation of decision>",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- If you see the final destination mentioned in instruction, consider STOP action
- If already very close to the goal object, choose STOP
- If still need to navigate, choose NAVIGATE and provide bounding box of next target
- Target description should be specific and clear
- Consider the instruction completion progress based on visited targets
- Use the progress analysis to inform your decision"""

VA_PROMPT_OBJECTNAV = """Object Navigation Task: "{instruction}"

Current View Status:
- Step: {current_step}
- Image size: {width}x{height} pixels{visited_targets_str}{progress_info}

Goal: Find and approach the "{target_object}".

Your Task:
1. **Detect**: Check if "{target_object}" is present in the image.
2. **Contextualize**: If not found, identify the current room type and visible exits (doors, hallways).
3. **Act**:
   - If **Found**: Set bbox around the "{target_object}" and NAVIGATE.
   - If **Searching**: Set bbox around the most promising area to explore (e.g., a door, a hallway, or a region likely to contain the target).
   - **STOP**: Only if you are within 1.0 meters of the "{target_object}".

Response format (JSON):
{{
    "progress": "<searching status>",
    "reasoning": "<logic for choice>",
    "action": "NAVIGATE" or "STOP",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of the selected target object or area>"
}}
"""

STOP_CHECK_PROMPT = """You are an intelligent navigation agent.
Task: "{instruction}"

You have decided to STOP, believing you have reached the target.
Now, please double-check your decision based on the current 4-directional views.

Current Views:
{current_views}

Requirements:
1. Check if the target object is clearly visible in any of the views.
2. Estimate if the distance to the target is less than 1 meter.

Decision Rules:
- If target is visible AND distance < 1m: CONFIRM STOP.
- If target is NOT visible OR distance >= 1m: CONTINUE NAVIGATION.

Response format (JSON):
{{
    "analysis": "<analysis of visibility and distance>",
    "decision": "STOP" or "CONTINUE"
}}
"""

GUIDELINES_GENERATOR = """You are given a natural language route instruction for an indoor navigation task.

Instruction:
"{instruction}"

Your task:
1. Read and fully understand the instruction.
2. Summarize it into stable, high-level navigation guidelines that should be remembered at EVERY step of the episode.

Requirements for the guidelines:
- Be concise (3–7 bullet points).
- Cover:
  - The global navigation goal (final destination / outcome).
  - Key intermediate landmarks or sub-goals to pass by.
  - Important turning patterns or overall direction trends.
  - Safety or motion constraints (e.g., stay close to wall, avoid stairs if not mentioned, etc.).
  - When to STOP (stopping condition) according to the instruction.
- Do NOT refer to step indices, images, or past actions.
- Use generic wording that remains valid throughout the whole episode.

Response format:
Return ONLY the guidelines as a numbered list, one guideline per line, for example: 
1. Exit the current room by turning around and passing through the doorway.  
2. Turn left and exit the next door encountered.  
3. Proceed towards the pool as your destination.  
4. Stay alert for obstacles and maintain a clear path.  
5. Stop when you reach the area next to the pool.  
6. Avoid entering any other doors or rooms unless instructed.  
7. Ensure you are positioned safely next to the pool before stopping. 
... (The guidelines will vary based on the instruction provided, this example is just for illustration of the format.)
"""

GUIDELINES_GENERATOR_OBJECTNAV = """You are an expert indoor navigation agent. Your task is to find a specific object in an unknown environment.

Target Object: "{target_object}"

Your task:
Generate 3-5 strategic guidelines for finding this object efficiently.

Requirements:
1. **Probable Locations**: Identify the room types where "{target_object}" is typically found (e.g., Bed -> Bedroom, Stove -> Kitchen, Sofa -> Living Room).
2. **Visual Cues**: List associated objects that might indicate proximity to the target (e.g., seeing a mirror might mean a sink/bathroom is near).
3. **Exploration Strategy**: Advise on how to explore (e.g., checking multiple rooms quickly vs. searching inside cabinets/corners).
4. **Stopping Condition**: Explicitly state to STOP when the object is clearly visible and within 1.0 meters.

Response format:
Return ONLY the numbered list of guidelines.
"""

# --- No-TODO variants (no_todo_prompt; used by the multi-agent LaViRA runner). ---
LA_PROMPT_BACKTRACK_NO_TODO = """Based on the navigation history and current 4-directional views,
decide the next action to find the target object.

{feedback}

Task:
- Decide the next action. Choose one of these actions:
{action_list}

Response format (JSON):
{{
    "map_understanding": "<if a team top-down map is provided: where YOU are by YOUR color in the legend, teammate pose/heading by their colors, FOUND from the text legend if any, N/S/E/W direction of next place (up=North); else empty string>",
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of the chosen action>",
    "action": "{action_desc}",
    "stop": True/False,
    "claim_goal_id": "goal_N or empty string"
}}

Guidelines:
- Do NOT try to open the doors.
- Stay on the current floor. Do not plan to go up or down stairs.
- If you believe you have reached the target object, set "stop": True.
- Focus on finding the target object specified in the task.
- When a team top-down map is present, fill map_understanding FIRST (find YOUR color in the legend, teammates by color, FOUND from the text legend if any, N/S/E/W where you will go; up=North). Colored X is only a short-term waypoint, not a FOUND object.
{negative_constraints}"""

LA_PROMPT_NO_BACKTRACK_NO_TODO = """Based on the navigation history and current 4-directional views,
decide the next navigation direction to find the target object.

{feedback}

Task:
- Decide the next action. Choose one of these actions:
{action_list}


Response format (JSON):
{{
    "map_understanding": "<if a team top-down map is provided: where YOU are by YOUR color in the legend, teammate pose/heading by their colors, FOUND from the text legend if any, N/S/E/W direction of next place (up=North); else empty string>",
    "progress_analysis": "<assessment of search progress>",
    "reasoning": "<explanation of chosen action>",
    "action": "{action_desc}",
    "stop": False,
    "claim_goal_id": "goal_N or empty string"
}}

Guidelines:
- Focus on finding the target object specified in the task.
- Stay on the current floor. Do not plan to go up or down stairs.
- When a team top-down map is present, fill map_understanding FIRST (find YOUR color in the legend, teammates by color, FOUND from the text legend if any, N/S/E/W where you will go; up=North). Colored X is only a short-term waypoint, not a FOUND object.
{negative_constraints}"""

VA_PROMPT_NO_TODO = """Navigation Task: "{instruction}"

Current situation:
- Step: {current_step}
- Image size: {width}x{height} pixels
- Progress Info: {progress_info}

Your task:
1. Identify the most relevant target object/area for what you should do next. Specify ONLY ONE. And it should not be too close to you.
2. Provide the bounding box of the target.

Response format (JSON):
{{
    "reasoning": "<brief explanation of decision>",
    "bbox_2d": [x1, y1, x2, y2],
    "target": "<description of target object>"
}}

Guidelines:
- Target description should be specific and clear.
- Stay on the current floor. Do not select stairs as the next target.
- Provide bounding box of next target."""
