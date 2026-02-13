# Secrets Management

This directory contains sensitive API keys and credentials for the Stock Analysis API.

## ⚠️ SECURITY WARNING

**NEVER commit these files to version control!** The `.gitignore` file is configured to exclude this directory.

## Required Secret Files

Create the following text files in this directory:

### 1. `gemini_api_key.txt`
Contains your Google Gemini API key (single line, no quotes).

```bash
echo "your_gemini_api_key_here" > gemini_api_key.txt
```

**How to get it:**
- Visit https://makersuite.google.com/app/apikey
- Create a new API key
- Copy and paste into this file

### 2. `serper_api_key.txt`
Contains your Serper API key for web search (single line, no quotes).

```bash
echo "your_serper_api_key_here" > serper_api_key.txt
```

**How to get it:**
- Visit https://serper.dev/
- Sign up for an account
- Copy your API key from the dashboard
- Paste into this file

### 3. `api_keys.txt`
Contains comma-separated list of valid API keys for client authentication (single line, no quotes).

```bash
echo "key1_abc123xyz,key2_def456uvw,key3_ghi789rst" > api_keys.txt
```

**How to generate secure keys:**
```bash
# Linux/Mac
openssl rand -hex 32

# Or use Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Windows PowerShell
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
```

**Best practices:**
- Generate at least one key per client application
- Use cryptographically secure random key generation
- Minimum 32 characters recommended
- Rotate keys periodically

## File Permissions

Set restrictive permissions on secret files:

```bash
# Linux/Mac
chmod 600 secrets/*.txt
chmod 700 secrets/

# Verify
ls -la secrets/
```

## Docker Compose Integration

These files are automatically mounted as Docker secrets in `docker-compose.yml`:

```yaml
secrets:
  gemini_api_key:
    file: ./secrets/gemini_api_key.txt
  serper_api_key:
    file: ./secrets/serper_api_key.txt
  api_keys:
    file: ./secrets/api_keys.txt
```

Docker secrets are:
- Mounted at `/run/secrets/<secret_name>` inside containers
- Only accessible to the service that needs them
- Not visible in container environment variables
- Not logged or exposed in `docker inspect`

## Reading Secrets in Code

The application reads secrets from Docker secrets:

```python
# src/core/config_validator.py
def read_secret(secret_name: str) -> str:
    secret_path = f"/run/secrets/{secret_name}"
    if os.path.exists(secret_path):
        with open(secret_path) as f:
            return f.read().strip()
    # Fallback to environment variable
    return os.getenv(secret_name.upper(), "")
```

## Development vs Production

**Development (without Docker):**
- Use `.env` file with environment variables
- Copy `.env.example` to `.env`
- Add your API keys directly

**Production (with Docker):**
- Use Docker secrets (this directory)
- More secure than environment variables
- No secrets stored in images or container configs

## Verification

Check that secrets are properly configured:

```bash
# 1. Verify files exist
ls -la secrets/

# 2. Check file contents (CAREFUL - this exposes secrets!)
cat secrets/gemini_api_key.txt
cat secrets/serper_api_key.txt
cat secrets/api_keys.txt

# 3. Test with Docker Compose
docker-compose up -d

# 4. Check logs for secret loading errors
docker-compose logs stock-api | grep -i "secret\|api.*key"

# 5. Test authentication
curl -H "X-API-Key: your_first_key_from_api_keys.txt" http://localhost:8000/api/stock/AAPL
```

## Troubleshooting

**"Secret not found" error:**
- Verify file exists: `ls secrets/gemini_api_key.txt`
- Check file path in `docker-compose.yml` is correct
- Ensure no extra whitespace or quotes in files

**"Invalid API key" error:**
- Check key format (no quotes, no newlines)
- Remove trailing spaces: `echo -n "key" > file.txt`
- Verify key is valid on provider's dashboard

**Permission denied:**
- Set correct permissions: `chmod 600 secrets/*.txt`
- Run Docker with appropriate user permissions

## Rotating Secrets

To rotate API keys:

1. **Generate new keys** on provider dashboards
2. **Update secret files** with new values
3. **Restart services**: `docker-compose restart stock-api`
4. **Verify** new keys work
5. **Revoke old keys** on provider dashboards

## Backup

**DO NOT** commit secrets to git repositories.

For secure backup:
- Use encrypted password managers (1Password, LastPass)
- Use cloud secret managers (AWS Secrets Manager, Azure Key Vault)
- Encrypt backups with GPG/age before storing

## Example Setup

```bash
# 1. Navigate to project directory
cd /path/to/stock-algorithms

# 2. Create secrets directory
mkdir -p secrets

# 3. Generate API keys on provider websites
# (Visit Gemini API and Serper websites)

# 4. Create secret files
echo "AIzaSyC_your_gemini_api_key_here" > secrets/gemini_api_key.txt
echo "your_serper_api_key_here" > secrets/serper_api_key.txt

# 5. Generate client API keys
python -c "import secrets; print(','.join([secrets.token_urlsafe(32) for _ in range(3)]))" > secrets/api_keys.txt

# 6. Set permissions
chmod 600 secrets/*.txt

# 7. Start services
docker-compose up -d

# 8. Test
curl http://localhost:8000/health
```

## .gitignore Configuration

Ensure `.gitignore` includes:

```
secrets/
secrets/*.txt
*.key
*.pem
.env
```

This prevents accidental commits of sensitive data.

---

**Questions or Issues?**
- Review docker-compose.yml for secret configuration
- Check logs: `docker-compose logs stock-api`
- Ensure API keys are valid on provider dashboards
