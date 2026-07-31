# 📊 Nexlink Telecom Database Entity-Relationship Diagram (ERD)

This document describes the schema architecture for the Nexlink Network Operations Center (NOC) SQLite database.

```mermaid
erDiagram
    users {
        int id PK
        string username UK
        string role "NOC_Admin | NOC_Engineer | Guest"
        string api_token UK
    }

    customers {
        int id PK
        string name
        string industry
        string sla_tier "VIP | Enterprise | Standard"
        string contact_email
    }

    network_nodes {
        int id PK
        string name
        string type "Fiber | 5G Core | Edge Router | Satellite"
        float max_capacity_gbps
        float current_load_gbps
        string status "Healthy | Congested | Down | Maintenance"
        string location
    }

    services {
        int id PK
        int customer_id FK
        int node_id FK
        float allocated_bandwidth_gbps
        string status "Active | Suspended | Pending"
    }

    audit_logs {
        int id PK
        datetime timestamp
        int user_id
        string action
        string details
    }

    customers ||--o{ services : "subscribes to"
    network_nodes ||--o{ services : "hosts"
    users ||--o{ audit_logs : "triggers"
```

## Entity Descriptions

1. **users**: Tracks system operators and their role-based access permissions (`NOC_Admin`, `NOC_Engineer`, `Guest`). Used by the MCP server for handler-level authorization and dynamic toolset changes (`tools/list_changed`).
2. **customers**: Enterprise and VIP clients subscribing to telecom infrastructure. The `sla_tier` field determines whether bandwidth modifications trigger mid-call protocol **Elicitation**.
3. **network_nodes**: Physical and virtual network nodes monitored by the NOC. Tracks operational status and real-time bandwidth loads.
4. **services**: Relational mapping linking customers to specific network nodes and allocated bandwidth quotas.
5. **audit_logs**: Immutable audit trail documenting state changes, bandwidth upgrades, and security approvals.
