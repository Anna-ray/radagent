# RadAgent MI300X Cost Playbook

DigitalOcean MI300X 1x plan: **$1.99/hr**. Total credit: **$100**.
Hard rule: never have a droplet running unattended. Always set a wall-clock alarm.

## Session budgets

| Session                          | Time  | Cost  | Cumulative |
|----------------------------------|-------|-------|------------|
| 1. Boot + bootstrap + 11B download | 2h    | $4.0  | $4         |
| 2. First vLLM smoke test         | 1h    | $2.0  | $6         |
| 3. End-to-end pipeline test (10 imgs) | 2h    | $4.0  | $10        |
| 4. Bench harness run (50 imgs, 11B) | 2h    | $4.0  | $14        |
| 5. Optional: 90B download + bench | 4h    | $8.0  | $22        |
| 6. Buffer for re-runs            | 8h    | $16   | $38        |

Target: stay under $40, leave $60 for the dashboard sessions or unexpected work.
**Hard ceiling: $80.** If the meter passes $80, stop everything, snapshot, and re-plan.

## Pre-session checklist (run BEFORE creating a droplet)

- [ ] Wall-clock alarm set on phone for `2h` (or whatever the session budget is)
- [ ] `HF_TOKEN` ready locally (env var)
- [ ] Local artifacts staged: specialist checkpoint, RAG index, sample images
- [ ] `infra/mi300x_bootstrap.sh` and `infra/run_vllm.sh` already written
- [ ] DO billing dashboard open in a browser tab as your meter

## During-session checklist

- [ ] First 5 min: confirm `rocm-smi` shows the GPU
- [ ] First 30 min: bootstrap script runs cleanly
- [ ] First 60 min: 11B download is finished (run `du -sh /workspace/hf_cache`)
- [ ] First 90 min: vLLM server responds to a curl health check
- [ ] At 80% of budget: stop new work, write down notes, snapshot if you want to resume
- [ ] At 100% of budget: **shutdown immediately**, regardless of state

## Post-session checklist (CRITICAL)

After every session, in this order:

1. `tmux kill-server` on the droplet (kills any background process)
2. SSH out
3. Open DO dashboard
4. **Destroy the droplet** (not "shut down" — destroy. shutdown still bills storage.)
5. Verify: refresh dashboard, no droplets listed under your project
6. Note actual cost in `infra/session_log.md` (create if missing)

## Auto-destroy safety net

Add this to your droplet `crontab` during bootstrap if you tend to forget:

    # Auto-shutdown after 4 hours, regardless of activity.
    # Replace 240 with your session budget in minutes.
    sudo shutdown -h +240

DigitalOcean still bills for halted droplets. `shutdown -h` will halt the OS;
you must still **destroy** in the dashboard. The cron only protects against
runaway processes, not against forgetting.

## Connecting

You already have an SSH key registered (`radagent-rayane`). When creating the
droplet on DO, select that key — DO injects it as `~/.ssh/authorized_keys`
on the droplet automatically.

    ssh -i ~/.ssh/radagent-rayane root@<droplet-ip>

If you get "Permission denied (publickey)", verify the key is added in DO
under Settings -> Security, AND that you selected it during droplet creation.

## What goes wrong (and how to recover)

- **vLLM OOM at startup**: lower `GPU_MEM_FRAC` to 0.75. MI300X has 192 GB so
  this only matters if you also load the specialist + RAG.
- **HF download hangs**: re-source `env.sh` (HF_HUB_ENABLE_HF_TRANSFER=1 is the
  fast path). If still slow, fall back to `git clone` of the model repo.
- **Token expired / wrong**: regenerate on huggingface.co/settings/tokens, re-export.
- **Bootstrap script fails on `pip install vllm`**: try `pip install vllm-rocm`
  or build from source (`pip install -e .` from a `git clone` of vLLM main).
