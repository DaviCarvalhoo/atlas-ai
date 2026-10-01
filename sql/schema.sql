-- Atlas operational schema (portable: SQLite + PostgreSQL)

CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT        NOT NULL,
    email         TEXT        NOT NULL,
    cpf           TEXT        NOT NULL,
    phone         TEXT        NOT NULL,
    state         CHAR(2)     NOT NULL,
    tier          TEXT        NOT NULL CHECK (tier IN ('standard', 'silver', 'gold')),
    signup_date   DATE        NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id            INTEGER PRIMARY KEY,
    customer_id         INTEGER NOT NULL REFERENCES customers (customer_id),
    created_at          TIMESTAMP NOT NULL,
    status              TEXT    NOT NULL,
    total_value         NUMERIC(10, 2) NOT NULL,
    payment_method      TEXT    NOT NULL,
    carrier             TEXT    NOT NULL,
    estimated_delivery  DATE    NOT NULL,
    delivered_at        DATE
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id         INTEGER PRIMARY KEY,
    customer_id       INTEGER NOT NULL REFERENCES customers (customer_id),
    order_id          INTEGER REFERENCES orders (order_id),
    created_at        TIMESTAMP NOT NULL,
    channel           TEXT    NOT NULL,
    subject           TEXT    NOT NULL,
    body              TEXT    NOT NULL,
    category          TEXT    NOT NULL,
    priority          TEXT    NOT NULL,
    status            TEXT    NOT NULL,
    resolution_hours  REAL,
    csat              INTEGER
);

CREATE INDEX IF NOT EXISTS idx_orders_customer   ON orders (customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_customer  ON tickets (customer_id);
CREATE INDEX IF NOT EXISTS idx_tickets_created   ON tickets (created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_category  ON tickets (category);

-- Read-only view exposed to the agent: no PII columns.
CREATE VIEW IF NOT EXISTS v_customer_safe AS
SELECT customer_id, state, tier, signup_date
FROM customers;
