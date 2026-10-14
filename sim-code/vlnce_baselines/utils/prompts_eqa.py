
LA_PROMPT_BACKTRACK = """Based on the navigation history, current 4-directional views, and the TODO list,
decide the next action to find the target object. If appropriate, suggest incremental updates to the TODO list.

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
     `index` to inject at a position; omit to append at end); `op:remove` to drop.
   - `completed` WITHOUT a concrete `result` will be rolled back.

2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO
   plus what you observe.
   Choose one of these actions:
{action_list}

Response format (JSON — strictly in this order):
{{
    "progress_analysis": "<brief assessment of what you've observed and done so far>",
    "reasoning_todo": "<reasoning behind the TODO updates>",
    "todo_updates": [
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
- Set stop=true whenever you have reached the target area needed to answer the question.
  The TODO list does NOT gate stopping — you may stop regardless of how many pending remain.
- Focus on finding the target object specified in the task.
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
{negative_constraints}"""


LA_PROMPT_BACKTRACK_NO_TODO = """Based on the navigation history and current 4-directional views, decide the next action to find the target object.

{feedback}

Task:
- Decide the next action. Choose one of these actions:
{action_list}

Response format (JSON):
{{
    "progress_analysis": "<assessment of current progress toward instruction completion>",
    "reasoning": "<explanation of the chosen action>",
    "action": "{action_desc}",
    "stop": True/False,
    "stair": "up"|"down"|False
}}

Guidelines:
- Do NOT try to open the doors.
- If you are going up/down stairs (or steps), indicate "stair": "up" or "down", otherwise False.
- If you believe you have reached the target object, set "stop": True.
- Focus on finding the target object specified in the task.
{negative_constraints}"""


LA_PROMPT_NO_BACKTRACK_NO_TODO = """Based on the navigation history and current 4-directional views, decide the next navigation direction to find the target object.

{feedback}

Task:
- Decide the next action. Choose one of these actions:
{action_list}

Response format (JSON):
{{
    "progress_analysis": "<assessment of search progress>",
    "reasoning": "<explanation of chosen action>",
    "action": "{action_desc}",
    "stop": False,
    "stair": "up"|"down"|False
}}

Guidelines:
- Focus on finding the target object specified in the task.
- If you are going up/down stairs (or steps), indicate "stair": "up" or "down", otherwise False
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
- If deciding to go up or down stairs, the bbox MUST select the ENTRY of the stairs.
- Provide bounding box of next target."""


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
