# AIGuard Streamlit Demo

Interactive web interface for testing AIGuard security guardrails.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for enhanced PII detection)
python -m spacy download en_core_web_sm
```

## Running the Demo

```bash
streamlit run src/demo/app.py
```

The demo will open in your browser at `http://localhost:8501`

## Features

### Main Testing Interface
- **Text Input**: Enter any prompt or message to test
- **Real-time Analysis**: Get instant security assessment
- **Visual Indicators**: Green (safe) or red (threat detected) badges
- **Detailed Results**: See exactly what was detected and why

### Guardrail Toggles
Enable/disable individual detection modules:
- ✅ Prompt Injection Detection
- ✅ Jailbreak Detection
- ✅ PII Detection & Redaction
- ✅ Encoding Attack Detection
- ✅ Output Filtering

### Adversarial Examples
Pre-loaded attack patterns to test:
- **Prompt Injection**: "Ignore all previous instructions..."
- **Jailbreaks**: DAN mode, developer mode, evil twin
- **Encoding Attacks**: Base64, hex, Unicode, ROT13
- **PII Examples**: SSN, credit cards, emails, phones
- **Safe Inputs**: Legitimate queries (should pass)

### Configuration Options
- Adjust detection thresholds
- Change PII redaction mode (full/partial/token/mask/hash)
- Provide your system prompt for leakage detection

## Example Usage

### Test a Prompt Injection Attack
1. Go to the "Adversarial Examples" tab
2. Click "Ignore All Previous"
3. Click "🔍 Analyze Input"
4. See detection results and threat type

### Test PII Redaction
1. Go to "Adversarial Examples" → "PII Examples"
2. Click "SSN" or "Email"
3. Analyze to see PII detection and redaction

### Compare Original vs Sanitized
When a threat is detected, the demo shows:
- **Original Input**: The text you entered
- **Sanitized Output**: The text with threats/PII removed

## Customization

### Adjust Thresholds
Use the sidebar to adjust:
- **Injection Similarity Threshold** (default: 0.75)
- **Jailbreak Threshold** (default: 0.80)

Lower = more sensitive (catches more threats, may increase false positives)
Higher = less sensitive (fewer false positives, may miss some threats)

### PII Redaction Modes
Choose how PII is redacted:
- **full**: `[SSN REDACTED]`
- **partial**: `XXX-XX-1234`
- **token**: `[EMAIL_1]`, `[PHONE_2]`
- **mask**: `****************`
- **hash**: `[SSN_HASH:4321]`

## Performance Notes

- **First Run**: Models download on first run (~500MB)
- **Subsequent Runs**: Models cached, much faster
- **CPU vs GPU**: Uses CPU by default; set `device=cuda` for GPU acceleration

## Tips

1. **Start with examples**: Use pre-loaded examples to understand the interface
2. **Test your own prompts**: Paste real user queries to test
3. **Adjust thresholds**: Fine-tune for your use case
4. **Check safe inputs**: Verify legitimate queries pass through

## Troubleshooting

### Port Already in Use
```bash
streamlit run src/demo/app.py --server.port 8502
```

### Models Not Downloading
Check your internet connection and try:
```bash
export HF_HOME=/path/to/cache
streamlit run src/demo/app.py
```

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

## Next Steps

- Integrate AIGuard into your FastAPI app using the middleware
- Run the full adversarial test suite
- Customize detection patterns for your use case
