# run everything with default model
# baseline first (replication anchor)
python -m agent_bullwhip.runner --config baseline --runs 30 --horizon 36
python -m agent_bullwhip.runner --config mirror --runs 30 --horizon 36
python -m agent_bullwhip.runner --config order_up_to --runs 30 --horizon 36

# inference-time reliability fixes (our extension)
python -m agent_bullwhip.runner --config voting5 --runs 30 --horizon 36
python -m agent_bullwhip.runner --config voting10 --runs 30 --horizon 36
python -m agent_bullwhip.runner --config guardrail --runs 30 --horizon 36
python -m agent_bullwhip.runner --config anchor --runs 30 --horizon 36
python -m agent_bullwhip.runner --config prompt_weighted --runs 30 --horizon 36
python -m agent_bullwhip.runner --config combined --runs 30 --horizon 36

# model ladder (optional)
# python -m agent_bullwhip.runner --config baseline --runs 30 --model glm-5.2
