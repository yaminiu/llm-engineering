# GitHub Copilot Instructions

This document provides guidance for GitHub Copilot when assisting with the Confluent Kubernetes POC repository.

## Repository Context

This repository contains infrastructure-as-code, Kubernetes configurations, and cloud-native applications for Confluent Cloud deployments.

### Key Areas
- **Kubernetes Manifests** (`*.yaml`, `*.yml`) - Helm charts, KRaft configurations, Kafka brokers
- **Infrastructure as Code** (`*.tf`) - Terraform configurations for cloud resources
- **Ansible Playbooks** (`cp_ansible-7.9.1/`) - Configuration management and deployment
- **Kafka Applications** (Python, Java) - Producers, consumers, streaming applications
- **Schema Management** (Python) - Avro schema handling
- **Monitoring** (Prometheus, Control Center) - Observability stack

## Core Principles

1. **Security First** - Always validate secrets handling, TLS configurations, and access control
2. **Infrastructure as Code** - Prefer declarative over imperative, maintain idempotency
3. **Documentation** - Keep deployment guides, architecture diagrams, and runbooks current
4. **Testing** - Validate configurations before production deployment
5. **Monitoring** - Include observability in all deployments

## Guidance by Task Type

### Planning Code Changes
**Agent**: Use `change-impact-planner` agent
- Trigger: "help me plan", "what's the impact", "analyze risks"
- Covers: Dependency tracing, risk assessment, test strategy

### Kubernetes Configuration
**Mode**: `confluent-cfk` 
- Expertise: K8s manifests, Helm charts, ConfigMaps, Secrets
- Best Practices: Labels, resource limits, health checks, node affinity
- Common Issues: Image pulls, secret mounts, networking

### Terraform & Infrastructure
**Mode**: `confluent-terraform`
- Expertise: Provider configs, state management, modules
- Best Practices: Variable abstraction, output exports, environment separation
- Common Issues: State locks, version constraints, credential management

### Deployment & Automation
**Mode**: `confluent-platform`
- Expertise: Ansible playbooks, deployment sequencing, cluster setup
- Best Practices: Idempotent plays, error handling, inventory management
- Common Issues: Version compatibility, credential handling, network requirements

### Documentation
**Mode**: `documentation`
- Expertise: Architecture guides, deployment runbooks, API documentation
- Best Practices: Clear examples, troubleshooting sections, version notes
- Common Issues: Outdated references, missing prerequisites

## Common Workflows

### "I need to deploy avatar-eventbus to Kubernetes"
1. Review `avatar-eventbus/` Helm chart structure
2. Check `k8s_tls_secret.yaml` for certificate setup
3. Validate `deploy-avatar-eventbus.yml` workflow
4. Use **confluent-cfk** mode for manifest review

### "I need to set up Kafka broker cluster"
1. Check `kafka_broker.yaml` configuration
2. Review `kafka_client.yaml` for client setup
3. Validate `kafka_producer_*.yaml` for producer configs
4. Use **confluent-cfk** mode for K8s guidance

### "I'm adding a new connector"
1. Reference existing `kafka-connector/` and `datagen-connector.yaml`
2. Check `connect_shr_proxy.yaml` for connector networking
3. Use **confluent-cfk** mode for manifest validation
4. Update documentation in `docs/`

### "I need to modify Terraform configs"
1. Review existing `.tf` files (if any)
2. Check `Confluent_Cloud/` directory structure
3. Use **confluent-terraform** mode for IaC guidance
4. Validate state management practices

## Environment Considerations

- **Minikube** - Used for local development (see `start_minikube.bash`)
- **TLS/mTLS** - Multiple TLS configs available:
  - `k8s_tls_secret.yaml` - Standard TLS
  - `kraft_mtl_autogen_secret.yaml` - Kraft with mTLS
  - `nonprod_tls_secrets.yaml` - Non-production secrets
- **License** - Confluent Enterprise license in `license_secrt.yaml`
- **Monitoring** - Prometheus stack in `promethues/` directory

## Code Style & Standards

- YAML: 2-space indentation, use `---` document separators
- Terraform: Follow HashiCorp conventions, include descriptions
- Python: PEP 8, include docstrings for main functions
- Kubernetes: Use labels, annotations, and namespace segregation
- Documentation: Markdown with code blocks and examples

## Security Checklist

When reviewing or creating configurations:
- [ ] Secrets managed via Kubernetes Secrets or HashiCorp Vault
- [ ] TLS/mTLS enabled for inter-service communication
- [ ] RBAC configured for Kubernetes resources
- [ ] Network policies defined (if applicable)
- [ ] Credentials not embedded in code or images
- [ ] Secret rotation strategy documented
- [ ] Audit logging enabled

## Quick References

| Topic | File/Directory |
|-------|---------------|
| Kafka Broker Config | `kafka_broker.yaml` |
| Client Setup | `kafka_client.yaml` |
| Helm Charts | `avatar-eventbus/` |
| TLS Secrets | `k8s_tls_secret.yaml`, `kraft_mtl_autogen_secret.yaml` |
| Ansible Automation | `cp_ansible-7.9.1/` |
| Monitoring Stack | `promethues/` |
| CI/CD Workflows | `.github/workflows/` |
| Local Development | `start_minikube.bash` |

## Getting Help

Use these phrases to trigger specific assistance:

- **"Help me plan [change]"** → Impact analysis and strategy
- **"How do I deploy..."** → Kubernetes/deployment guidance
- **"Optimize this Terraform..."** → Infrastructure review
- **"Write documentation for..."** → Documentation mode
- **"What's the risk of..."** → Impact and risk analysis
- **"Fix this K8s manifest..."** → Kubernetes best practices
