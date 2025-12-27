# Finance Agent Benchmark - Deployment Quick Start

## ✅ Repository Setup: DONE
Repository: `git@github.com:oedemis/finance-agent-evaluator.git`

---

## Step 1: Add GitHub Secrets (5 minutes)

Go to: **https://github.com/oedemis/finance-agent-evaluator/settings/secrets/actions**

Click **"New repository secret"** und füge hinzu:

### Required Secrets:
1. **SEC_EDGAR_API_KEY**
   - Value: Your SEC API key
   - Used for EDGAR filings search

2. **SERPAPI_API_KEY**
   - Value: Your SerpAPI key
   - Used for Google web searches

3. **OPENAI_API_KEY**
   - Value: Your OpenAI API key
   - Used for LLM judges (gpt-4o-mini)

---

## Step 2: Push Workflow to GitHub

```bash
cd /Users/ooedemis/dev/benchmarks/finance-agent-evaluator

# Add the workflow file
git add .github/workflows/publish.yml

# Commit
git commit -m "ci: add GitHub Actions workflow for Docker publishing"

# Push to GitHub
git push origin main
```

**Was passiert dann:**
- GitHub Actions startet automatisch
- Docker Image wird gebaut
- Tests werden ausgeführt (health check)
- Image wird zu GHCR gepusht

---

## Step 3: Check GitHub Actions

Go to: **https://github.com/oedemis/finance-agent-evaluator/actions**

Du solltest sehen:
- ✅ Workflow läuft
- ⏱️ Build dauert ~3-5 Minuten
- ✅ Docker Image wird gepusht

---

## Step 4: Verify Docker Image

Nach erfolgreichem Build:

**Image URL:**
```
ghcr.io/oedemis/finance-agent-evaluator:main
```

**Make it public:**
1. Go to: https://github.com/oedemis?tab=packages
2. Find: `finance-agent-evaluator`
3. Click package → Package settings
4. Scroll down → **Change visibility** → **Public**

---

## Step 5: Test Locally (Optional)

```bash
# Pull the published image
docker pull ghcr.io/oedemis/finance-agent-evaluator:main

# Run it
docker run -d --name fab-evaluator -p 9009:9009 \
  -e SEC_EDGAR_API_KEY="your_key" \
  -e SERPAPI_API_KEY="your_key" \
  -e OPENAI_API_KEY="your_key" \
  ghcr.io/oedemis/finance-agent-evaluator:main

# Test health
curl http://localhost:9009/health

# Test agent card
curl http://localhost:9009/ | jq

# Cleanup
docker stop fab-evaluator && docker rm fab-evaluator
```

---

## Step 6: Register on AgentBeats

Go to: **https://agentbeats.dev**

1. Login with GitHub
2. Click **"Register Agent"**
3. Fill in:
   - **Name**: `Finance Agent Benchmark`
   - **Docker Image**: `ghcr.io/oedemis/finance-agent-evaluator:main`
   - **Port**: `9009`
   - **Description**:
     ```
     Finance Agent Benchmark 2.0 - Evaluates agents on 537 real-world
     financial research tasks. $2M competition with multi-dimensional
     scoring: factual accuracy, contradiction detection, and process quality.
     ```

4. **IMPORTANT:** Copy your **Agent ID** (e.g., `agent_abc123xyz`)

---

## Step 7: Create Leaderboard Repository

1. Go to: https://github.com/RDI-Foundation/agentbeats-leaderboard-template
2. Click **"Use this template"** → **"Create a new repository"**
3. Name: `finance-agent-leaderboard`
4. Visibility: **PUBLIC**
5. Create repository

### Enable Workflow Permissions:
Go to: **Settings → Actions → General → Workflow permissions**
- Select: ✅ **"Read and write permissions"**
- Save

### Add Secrets to Leaderboard Repo:
Go to: **https://github.com/oedemis/finance-agent-leaderboard/settings/secrets/actions**

Add the same secrets:
- `SEC_EDGAR_API_KEY`
- `SERPAPI_API_KEY`
- `OPENAI_API_KEY`

---

## Step 8: Configure scenario.toml

Edit `scenario.toml` in your leaderboard repo:

```toml
[green_agent]
# Your Agent ID from AgentBeats (Step 6)
agentbeats_id = "agent_abc123xyz"  # REPLACE with your actual ID

# Environment variables (using GitHub Secrets)
env = {
    SEC_EDGAR_API_KEY = "${SEC_EDGAR_API_KEY}",
    SERPAPI_API_KEY = "${SERPAPI_API_KEY}",
    OPENAI_API_KEY = "${OPENAI_API_KEY}"
}

# Purple agent (participant) - left empty for submitters
[[participants]]
name = "agent"
agentbeats_id = ""  # Participants fill this
env = {}

# Benchmark configuration
[config]
num_tasks = 10
categories = ["all"]
max_steps = 50
timeout = 600
```

Push changes:
```bash
git add scenario.toml
git commit -m "feat: configure Finance Agent Benchmark"
git push origin main
```

---

## Step 9: Connect Leaderboard to AgentBeats

1. Go to: **https://agentbeats.dev** → Your agent
2. Click **"Edit Agent"**
3. **Leaderboard Repository**: `https://github.com/oedemis/finance-agent-leaderboard`
4. **Scoring Query** (DuckDB):
   ```sql
   SELECT
       submission_id,
       agent_name,
       AVG(class_balanced_accuracy) as score,
       AVG(benchmark_reward) as benchmark_score,
       AVG(process_quality) as process_quality,
       AVG(total_cost) as avg_cost_usd,
       COUNT(*) as num_runs
   FROM read_json('results/**/*.json')
   GROUP BY submission_id, agent_name
   ORDER BY score DESC
   ```
5. Save

### Setup Webhook:
1. Copy webhook URL from AgentBeats (in "Webhook Integration" box)
2. Go to: **https://github.com/oedemis/finance-agent-leaderboard/settings/hooks**
3. Click **"Add webhook"**
4. **Payload URL**: Paste webhook URL
5. **Content type**: `application/json`
6. **Events**: ✅ Just the push event
7. **Active**: ✅
8. Add webhook

---

## Step 10: Test End-to-End

### Create a test submission:

1. Fork your leaderboard repo (to simulate a participant)
2. Edit `scenario.toml`:
   ```toml
   [[participants]]
   name = "agent"
   agentbeats_id = "test_agent_baseline"  # Your baseline agent ID
   env = {}
   ```
3. Push changes
4. GitHub Actions automatically runs benchmark
5. Check results in `results/` directory
6. Create PR back to main repo

---

## Next Steps

✅ **Docker image published**: `ghcr.io/oedemis/finance-agent-evaluator:main`
⏳ **Register on AgentBeats**: https://agentbeats.dev
⏳ **Create leaderboard repo**
⏳ **Connect to AgentBeats**
⏳ **Test submission flow**
🚀 **Launch $2M competition!**

---

## Versioned Releases

Create versioned releases for stability:

```bash
# Tag a release
git tag v1.0.0
git push origin v1.0.0

# Creates image: ghcr.io/oedemis/finance-agent-evaluator:1.0.0
```

Update AgentBeats to use: `ghcr.io/oedemis/finance-agent-evaluator:1.0.0`

---

## Troubleshooting

### GitHub Actions Failed
- Check: https://github.com/oedemis/finance-agent-evaluator/actions
- Verify all 3 secrets are added correctly
- Check Docker build logs

### Docker Image Not Found
- Make package public: https://github.com/oedemis?tab=packages
- Verify workflow completed successfully

### Health Check Failed
- Check if port 9009 is exposed in Dockerfile
- Verify server.py runs with --host 0.0.0.0

### AgentBeats Can't Pull Image
- Ensure package is PUBLIC
- Test: `docker pull ghcr.io/oedemis/finance-agent-evaluator:main`
