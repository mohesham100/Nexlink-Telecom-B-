import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'nexlink.db')

def init_db():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS audit_logs;
    DROP TABLE IF EXISTS services;
    DROP TABLE IF EXISTS network_nodes;
    DROP TABLE IF EXISTS users;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        industry TEXT NOT NULL,
        sla_tier TEXT CHECK(sla_tier IN ('VIP', 'Enterprise', 'Standard')) NOT NULL,
        contact_email TEXT NOT NULL
    );

    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        role TEXT CHECK(role IN ('NOC_Admin', 'NOC_Engineer', 'Guest')) NOT NULL,
        api_token TEXT NOT NULL UNIQUE
    );

    CREATE TABLE network_nodes (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT CHECK(type IN ('Fiber', '5G Core', 'Edge Router', 'Satellite')) NOT NULL,
        max_capacity_gbps REAL NOT NULL,
        current_load_gbps REAL NOT NULL,
        status TEXT CHECK(status IN ('Healthy', 'Congested', 'Down', 'Maintenance')) NOT NULL,
        location TEXT NOT NULL
    );

    CREATE TABLE services (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        node_id INTEGER NOT NULL,
        allocated_bandwidth_gbps REAL NOT NULL,
        status TEXT CHECK(status IN ('Active', 'Suspended', 'Pending')) NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        FOREIGN KEY (node_id) REFERENCES network_nodes(id)
    );

    CREATE TABLE audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER,
        action TEXT NOT NULL,
        details TEXT NOT NULL
    );

    INSERT INTO users (id, username, role, api_token) VALUES
    (1, 'sarah_admin', 'NOC_Admin', 'token-admin-9988'),
    (2, 'alex_engineer', 'NOC_Engineer', 'token-eng-5544'),
    (3, 'guest_user', 'Guest', 'token-guest-0000');

    INSERT INTO customers (id, name, industry, sla_tier, contact_email) VALUES
    (405, 'Bank of Alexandria', 'Banking', 'VIP', 'noc@bankofalex.com'),
    (406, 'TechHub Solutions', 'Software', 'Enterprise', 'ops@techhub.io'),
    (407, 'Alexandria Hospital', 'Healthcare', 'VIP', 'it@alexhospital.org'),
    (408, 'Delta Logistics', 'Transport', 'Standard', 'support@deltalogistics.com');

    INSERT INTO network_nodes (id, name, type, max_capacity_gbps, current_load_gbps, status, location) VALUES
    (10, 'Alex-Fiber-Main', 'Fiber', 20.0, 19.5, 'Congested', 'Alexandria Datacenter A'),
    (11, 'Cairo-Fiber-East', 'Fiber', 50.0, 12.3, 'Healthy', 'Cairo East Hub'),
    (12, 'Alex-Fiber-North', 'Fiber', 10.0, 0.0, 'Down', 'Alexandria North Switch'),
    (13, 'Suez-5G-Core', '5G Core', 30.0, 5.0, 'Maintenance', 'Suez Station');

    INSERT INTO services (id, customer_id, node_id, allocated_bandwidth_gbps, status) VALUES
    (1, 405, 10, 2.0, 'Active'),
    (2, 406, 11, 1.0, 'Active'),
    (3, 407, 12, 5.0, 'Active'),
    (4, 408, 13, 0.5, 'Suspended');
    """)

    conn.commit()
    conn.close()
    print("Database setup complete.")

if __name__ == "__main__":
    init_db()