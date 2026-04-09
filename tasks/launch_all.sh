#!/bin/bash
cd /work/video_voice_ai_manager

claude --dangerously-skip-permissions -p "$(cat tasks/A1_core_modules.md)" > tasks/A1_output.log 2>&1 &
claude --dangerously-skip-permissions -p "$(cat tasks/A2_output_cli.md)" > tasks/A2_output.log 2>&1 &
claude --dangerously-skip-permissions -p "$(cat tasks/A3_web_setup.md)" > tasks/A3_output.log 2>&1 &

echo "3 agents launched, PIDs: $(jobs -p)"
wait
echo "All done"
