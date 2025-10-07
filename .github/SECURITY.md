# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Lexora seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do Not

- **Do not** open a public GitHub issue for security vulnerabilities
- **Do not** disclose the vulnerability publicly until it has been addressed

### Please Do

**Report security vulnerabilities to:** vesperakshay@gmail.com

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Communication**: We will keep you informed about the progress of fixing the vulnerability
- **Timeline**: We aim to address critical vulnerabilities within 7 days
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using Lexora, please follow these security best practices:

### API Keys and Secrets

- **Never commit API keys** to version control
- Use environment variables for sensitive configuration
- Rotate API keys regularly
- Use separate keys for development and production

```python
# ✅ Good - Use environment variables
import os
from lexora import RAGAgent, LLMConfig

agent = RAGAgent(
    llm_config=LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

# ❌ Bad - Hardcoded API key
agent = RAGAgent(
    llm_config=LLMConfig(
        api_key="sk-1234567890abcdef"  # Never do this!
    )
)
```

### Data Security

- **Sanitize user inputs** before processing
- **Validate data** before storing in vector databases
- **Encrypt sensitive data** at rest and in transit
- **Implement access controls** for production deployments

### Vector Database Security

- Use authentication for vector database connections
- Restrict network access to vector databases
- Regularly backup your vector database data
- Monitor for unusual access patterns

### LLM Security

- Be aware of prompt injection risks
- Validate and sanitize user queries
- Implement rate limiting for API calls
- Monitor LLM usage and costs

### Deployment Security

- Use HTTPS for all API endpoints
- Implement proper authentication and authorization
- Keep dependencies up to date
- Use security scanning tools in CI/CD

## Known Security Considerations

### Prompt Injection

RAG systems can be vulnerable to prompt injection attacks. Always:

- Validate and sanitize user inputs
- Use system prompts to set boundaries
- Implement content filtering
- Monitor for suspicious patterns

### Data Privacy

When using Lexora with sensitive data:

- Consider using local embeddings (HuggingFace) instead of cloud APIs
- Use local vector databases (FAISS) for sensitive data
- Implement data retention policies
- Comply with relevant data protection regulations (GDPR, CCPA, etc.)

### Dependency Security

We regularly update dependencies to address security vulnerabilities. To stay secure:

```bash
# Update Lexora to the latest version
pip install --upgrade lexora

# Check for security vulnerabilities in dependencies
pip install safety
safety check
```

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1, 0.1.2) and announced via:

- GitHub Security Advisories
- Release notes
- Email to security@lexora (if you've subscribed)

## Vulnerability Disclosure Policy

We follow responsible disclosure practices:

1. **Private Disclosure**: Report vulnerabilities privately first
2. **Fix Development**: We develop and test a fix
3. **Coordinated Release**: We coordinate the release with the reporter
4. **Public Disclosure**: After the fix is released, we publish a security advisory

## Security Hall of Fame

We recognize security researchers who help keep Lexora secure:

<!-- Security researchers will be listed here -->

*No vulnerabilities reported yet*

## Contact

For security-related questions or concerns:

- **Email**: vesperakshay@gmail.com
- **Subject**: [SECURITY] Your subject here

For general questions, please use [GitHub Discussions](https://github.com/VesperAkshay/lexora/discussions).

---

Thank you for helping keep Lexora and our users safe! 🔒
