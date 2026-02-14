# Troubleshooting Guide

Common issues and solutions for the Enterprise Call Intelligence Platform.

---

## Database Errors

### Error: "Cannot alter entry 'calls' because there are entries that depend on it"

**Cause:** Database already exists from a previous run with conflicting schema.

**Solution:** Clean up and start fresh:

```bash
# Option 1: Use make command
make clean

# Option 2: Manual cleanup
rm -f storage/structured/*.db
rm -rf storage/vectors storage/cache storage/search
mkdir -p storage/vectors storage/cache storage/search
```

Then run the pipeline again:
```bash
./run_pipeline.sh --num-calls 100 --skip-enrichment
```

---

## Import Errors

### Error: "ModuleNotFoundError: No module named 'pipelines'"

**Cause:** Python can't find the project modules.

**Solution:** Use the provided convenience scripts or set PYTHONPATH:

```bash
# Option 1: Use scripts (RECOMMENDED)
./run_pipeline.sh --num-calls 100

# Option 2: Set PYTHONPATH
PYTHONPATH=. python pipelines/orchestrator.py --num-calls 100

# Option 3: Use make commands
make run-pipeline-fast
```

### Error: "NameError: name 'Optional' is not defined"

**Cause:** Missing import statement.

**Solution:** This should be fixed in the latest code. If you still see it, ensure you have the latest version:

```bash
git pull  # If using git
# Or re-download the files
```

---

## Virtual Environment Issues

### Error: "Command 'python' not found"

**Cause:** Virtual environment not activated.

**Solution:**

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Verify activation (should show venv path)
which python
```

### Error: "No module named 'fastapi'" (or other dependencies)

**Cause:** Dependencies not installed.

**Solution:**

```bash
# Ensure venv is activated, then:
pip install -r requirements.txt

# If that fails, try upgrading pip first:
pip install --upgrade pip
pip install -r requirements.txt
```

---

## API/UI Startup Issues

### Error: "Address already in use" (Port 8000 or 8501)

**Cause:** Another process is using the port.

**Solution:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process (replace PID with actual process ID)
kill -9 PID

# Or use different port:
uvicorn api.main:app --port 8001
streamlit run ui/streamlit_app.py --server.port 8502
```

---

## Data Generation Issues

### Error: "FileNotFoundError" when generating data

**Cause:** Storage directories don't exist.

**Solution:**

```bash
# Create directories
mkdir -p storage/raw_transcripts
mkdir -p storage/structured
mkdir -p storage/vectors
mkdir -p storage/cache
mkdir -p storage/search
mkdir -p logs

# Or use make:
make setup
```

---

## LLM/API Key Issues

### Error: "OpenAI API key not found"

**Cause:** OPENAI_API_KEY not set.

**Solution:**

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your key
nano .env
# Add: OPENAI_API_KEY=sk-your-key-here

# Or skip LLM features:
./run_pipeline.sh --num-calls 100 --skip-enrichment
```

### Error: "Rate limit exceeded"

**Cause:** Too many LLM requests too quickly.

**Solution:**

1. Use `--skip-enrichment` flag for testing
2. Reduce number of calls: `--num-calls 50`
3. Wait a few minutes and try again
4. Check your OpenAI rate limits

---

## Performance Issues

### Pipeline taking too long

**Solutions:**

1. **Use fewer calls for testing:**
   ```bash
   ./run_pipeline.sh --num-calls 50 --skip-enrichment
   ```

2. **Skip enrichment** (saves 80% of time):
   ```bash
   --skip-enrichment
   ```

3. **Close other applications** to free up memory

### Out of memory errors

**Solutions:**

1. Use fewer calls: `--num-calls 50`
2. Close other applications
3. Increase system swap space
4. Use cloud instance with more RAM

---

## Query Issues

### UI shows "No results found"

**Cause:** Indices not built yet.

**Solution:**

```bash
# Run the full pipeline first
./run_pipeline.sh --num-calls 100 --skip-enrichment

# Then start UI
./run_ui.sh
```

### Queries timing out

**Cause:** Database not indexed or too complex query.

**Solution:**

1. Check database has data:
   ```bash
   sqlite3 storage/structured/calls.db "SELECT COUNT(*) FROM calls;"
   ```

2. Try simpler query first:
   - "How many calls?" (fast)
   - "Show me trends" (medium)
   - Complex semantic queries (slower)

---

## Docker Issues

### Error: "Cannot connect to Docker daemon"

**Cause:** Docker not running.

**Solution:**

```bash
# Start Docker Desktop (macOS/Windows)
# Or start Docker service (Linux):
sudo systemctl start docker
```

### Error: "Port already in use" in Docker

**Solution:**

```bash
# Stop existing containers
docker-compose down

# Remove all containers
docker-compose down --volumes

# Restart
docker-compose up
```

---

## General Debugging

### Enable verbose logging

```bash
# Set debug mode in .env
DEBUG=true
LOG_LEVEL=DEBUG

# Run pipeline
./run_pipeline.sh --num-calls 10
```

### Check logs

```bash
# View logs
tail -f logs/app.log

# Check for errors
grep ERROR logs/app.log
```

### Test individual services

```bash
# Test data generation only
PYTHONPATH=. python pipelines/data_generator.py --num-calls 10

# Test ingestion only
PYTHONPATH=. python services/ingestion_service.py

# Test preprocessing only
PYTHONPATH=. python services/preprocessing_service.py
```

---

## Still Having Issues?

1. **Check system requirements:**
   - Python 3.10+
   - 2GB+ RAM
   - 1GB+ free disk space

2. **Verify installation:**
   ```bash
   python --version  # Should be 3.10+
   pip list | grep fastapi  # Should show fastapi
   ```

3. **Start with minimal setup:**
   ```bash
   make clean
   ./run_pipeline.sh --num-calls 10 --skip-enrichment
   ```

4. **Check documentation:**
   - `README.md` - Overview
   - `docs/QUICKSTART.md` - Setup guide
   - `docs/ARCHITECTURE.md` - System design

5. **Common fixes:**
   ```bash
   # Nuclear option - complete reset
   make clean
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ./run_pipeline.sh --num-calls 10 --skip-enrichment
   ```

---

## Contact & Support

For additional help:
- Review code comments (heavily documented)
- Check error messages carefully
- Try minimal reproduction case
- Search for similar errors online

Most issues are due to:
1. Missing dependencies
2. Incorrect Python path
3. Port conflicts
4. Database state conflicts

The solutions above should resolve 95% of issues!
