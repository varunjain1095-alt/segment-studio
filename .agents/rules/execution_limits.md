# Execution & Command Approval Rules

## Strict Prohibition on Unprompted Long-Running Commands
1. **Explicit Permission Required**: The agent is strictly prohibited from running any long-running command, background process, batch data processing, model training, or database seeding without explicit user consent.
2. **Prior Justification & Estimation**: Before proposing or executing any command or script expected to take more than 5 seconds or consume non-trivial resources/tokens:
   - The agent MUST explicitly state the estimated execution time (duration).
   - The agent MUST explain why the command is actually required or whether it is optional.
   - The agent MUST stop and wait for the user's explicit confirmation before executing.
3. **No Polling Loops**: The agent must not enter loops polling or checking statuses that consume agent turns and user tokens.
4. **Immediate Direct Diagnosis**: When the user asks investigatory questions ("why is this happening?", "what went wrong?"), the agent must inspect files or read state directly and answer immediately without attempting automated repairs or pipeline runs.
