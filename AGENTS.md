# Workspace Rules & Security Limits

## Long-Running Commands & Execution Policy
- **No Autonomous Long-Running Tasks**: Never launch long-running scripts, background tasks, data pipelines, model training, or database seeding automatically.
- **Mandatory Disclosures Before Running Any Non-Trivial Task**:
  1. State the estimated execution time (duration).
  2. State why the task is required and whether it is strictly necessary or optional.
  3. Wait for the user's explicit approval before proceeding.
- **Token Conservation**: Never poll tasks in loops or chain unnecessary agent turns.
- **Investigatory Queries**: Answer diagnostic and explanatory questions directly without executing repair pipelines.
