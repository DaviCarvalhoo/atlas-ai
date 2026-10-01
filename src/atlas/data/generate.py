"""Seeded synthetic dataset for AtlasShop customer support.

Produces three related tables (customers, orders, tickets) with realistic properties:
Portuguese ticket text built from templates, ambiguous messages, ~5% label noise,
weekly seasonality, and labelled *incident days* (volume spikes) for anomaly detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

CATEGORIES = ["billing", "shipping", "technical", "account", "returns"]
PRIORITIES = ["low", "medium", "high", "urgent"]

FIRST_NAMES = ["Ana", "Bruno", "Carla", "Diego", "Eduarda", "Felipe", "Gabriela", "Heitor",
               "Isabela", "João", "Larissa", "Marcos", "Natália", "Otávio", "Paula", "Rafael",
               "Sofia", "Thiago", "Vitória", "Wagner"]
LAST_NAMES = ["Silva", "Santos", "Oliveira", "Souza", "Lima", "Pereira", "Costa", "Almeida",
              "Ferreira", "Rodrigues", "Gomes", "Martins"]
STATES = ["SP", "RJ", "MG", "PR", "RS", "SC", "BA", "PE", "GO", "DF", "CE", "ES"]
CARRIERS = ["Correios", "Jadlog", "Loggi", "Total Express"]
PAYMENTS = ["credit_card", "pix", "boleto", "debit_card"]
CHANNELS = ["email", "chat", "whatsapp", "phone"]
CARRIER_LATE_RATE = {"Correios": 0.24, "Jadlog": 0.15, "Loggi": 0.09, "Total Express": 0.17}
# Mean resolution hours per priority (urgent tickets are worked first) and SLA targets.
RESOLUTION_MEAN_H = {"urgent": 6.0, "high": 18.0, "medium": 30.0, "low": 55.0}
SLA_TARGET_H = {"urgent": 8, "high": 24, "medium": 48, "low": 120}

TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "Fui cobrado duas vezes no cartão pelo pedido {order}",
        "A cobrança do pedido {order} veio com valor diferente do anunciado",
        "Paguei o boleto há {days} dias e o pedido {order} continua aguardando pagamento",
        "Quero o estorno da compra {order}, o valor ainda aparece na fatura",
        "O pix do pedido {order} foi debitado mas não foi confirmado",
        "Minha fatura mostra uma cobrança que não reconheço",
        "O cupom de desconto não foi aplicado e paguei o valor cheio",
        "Preciso da nota fiscal do pedido {order}",
        "Parcelamento saiu em mais vezes do que escolhi",
    ],
    "shipping": [
        "Meu pedido {order} está atrasado há {days} dias",
        "O rastreio do pedido {order} não atualiza desde semana passada",
        "A transportadora {carrier} disse que entregou mas não recebi nada",
        "Onde está meu pedido {order}? Prazo já venceu",
        "O pedido {order} chegou com a caixa amassada e faltando item",
        "Quero alterar o endereço de entrega do pedido {order}",
        "Entrega marcada como devolvida ao remetente sem motivo",
        "Qual o prazo de entrega para {state}?",
        "O entregador da {carrier} não passou no horário combinado",
    ],
    "technical": [
        "O aplicativo fecha sozinho quando tento abrir o carrinho",
        "Não consigo finalizar a compra, aparece erro 500 no checkout",
        "A página de pagamento fica carregando infinitamente",
        "O app não mostra meus pedidos, tela em branco",
        "Erro ao aplicar cupom: mensagem de falha no sistema",
        "O site está muito lento e não carrega as fotos dos produtos",
        "Recebo erro ao tentar avaliar o produto do pedido {order}",
        "As notificações do app pararam de funcionar",
        "O botão de comprar não responde no navegador",
    ],
    "account": [
        "Não consigo fazer login, diz que a senha está errada",
        "Quero alterar o e-mail cadastrado na minha conta",
        "Minha conta foi bloqueada sem explicação",
        "Não recebo o código de verificação por SMS",
        "Quero excluir minha conta e meus dados pessoais (LGPD)",
        "Alguém acessou minha conta e mudou meu endereço",
        "Como atualizo meu CPF no cadastro?",
        "Esqueci minha senha e o link de redefinição expirou",
        "Quero sair da lista de e-mails promocionais",
    ],
    "returns": [
        "Quero devolver o produto do pedido {order}, não serviu",
        "Como faço a troca do item do pedido {order} por outro tamanho?",
        "O produto veio com defeito, quero trocar",
        "Solicitei a devolução há {days} dias e ninguém retirou o produto",
        "Me arrependi da compra {order}, quero cancelar e devolver",
        "Recebi o produto errado no pedido {order}",
        "A etiqueta de devolução não foi enviada",
        "Qual o prazo para trocar um produto?",
        "Quero cancelar o pedido {order} antes do envio",
    ],
}

# Generic sentences that appear across categories, making the task realistically ambiguous.
AMBIGUOUS = [
    "Preciso de ajuda com meu pedido {order}",
    "Ninguém responde meus e-mails, estou insatisfeito",
    "Péssimo atendimento, quero uma solução",
    "Podem verificar o que aconteceu com minha compra?",
]

OPENERS = ["", "", "Olá, ", "Bom dia, ", "Boa tarde. ", "Oi! ", "Prezados, "]
URGENCY = [" É urgente!", " Vou abrir reclamação no Procon.", " Isso é um absurdo, preciso resolver hoje."]
CLOSERS = ["", "", " Obrigado.", " Aguardo retorno.", " Att."]

# Incident scenarios: (category affected, multiplier, label)
INCIDENTS = [
    ("billing", 4.0, "payment gateway outage"),
    ("technical", 5.0, "checkout API down"),
    ("shipping", 3.5, "carrier strike"),
    ("account", 4.0, "SMS provider failure"),
]


@dataclass(frozen=True)
class SyntheticDataset:
    customers: pd.DataFrame
    orders: pd.DataFrame
    tickets: pd.DataFrame
    incidents: pd.DataFrame


def _cpf(rng: np.random.Generator) -> str:
    d = rng.integers(0, 10, size=11)
    return f"{d[0]}{d[1]}{d[2]}.{d[3]}{d[4]}{d[5]}.{d[6]}{d[7]}{d[8]}-{d[9]}{d[10]}"


def _phone(rng: np.random.Generator) -> str:
    return f"(11) 9{rng.integers(1000, 9999)}-{rng.integers(1000, 9999)}"


def _customers(rng: np.random.Generator, n: int, start: date) -> pd.DataFrame:
    rows = []
    for cid in range(1, n + 1):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        rows.append({
            "customer_id": cid,
            "name": f"{first} {last}",
            "email": f"{first.lower()}.{last.lower()}{cid}@example.com",
            "cpf": _cpf(rng),
            "phone": _phone(rng),
            "state": rng.choice(STATES),
            "tier": rng.choice(["standard", "silver", "gold"], p=[0.7, 0.2, 0.1]),
            "signup_date": start - timedelta(days=int(rng.integers(0, 900))),
        })
    return pd.DataFrame(rows)


def _orders(rng: np.random.Generator, customers: pd.DataFrame, n: int, start: date,
            days: int) -> pd.DataFrame:
    rows = []
    for oid in range(1, n + 1):
        created = datetime.combine(start, datetime.min.time()) + timedelta(
            days=int(rng.integers(0, days)), minutes=int(rng.integers(0, 1440)))
        eta = created.date() + timedelta(days=int(rng.integers(3, 12)))
        status = rng.choice(["delivered", "shipped", "processing", "cancelled", "returned"],
                            p=[0.68, 0.15, 0.08, 0.05, 0.04])
        carrier = str(rng.choice(CARRIERS))
        late = rng.random() < CARRIER_LATE_RATE[carrier]
        delay = int(rng.choice([1, 2, 3, 5, 9])) if late else 0
        rows.append({
            "order_id": 10000 + oid,
            "customer_id": int(rng.integers(1, len(customers) + 1)),
            "created_at": created,
            "status": status,
            "total_value": round(float(rng.lognormal(5.0, 0.7)), 2),
            "payment_method": rng.choice(PAYMENTS, p=[0.5, 0.3, 0.1, 0.1]),
            "carrier": carrier,
            "estimated_delivery": eta,
            "delivered_at": eta + timedelta(days=delay) if status == "delivered" else None,
        })
    return pd.DataFrame(rows)


def _ticket_text(rng: np.random.Generator, category: str, order_id: int, carrier: str,
                 state: str, pii: dict[str, str]) -> tuple[str, bool]:
    pool = TEMPLATES[category] + (AMBIGUOUS if rng.random() < 0.08 else [])
    body = rng.choice(pool).format(order=order_id, days=int(rng.integers(2, 20)),
                                   carrier=carrier, state=state)
    urgent = rng.random() < 0.12
    text = rng.choice(OPENERS) + body + (rng.choice(URGENCY) if urgent else "") + rng.choice(CLOSERS)
    # ~15% of customers paste personal data into the message (must be masked before any LLM).
    if rng.random() < 0.15:
        key = rng.choice(["email", "cpf", "phone"])
        text += f" Meu {'CPF' if key == 'cpf' else key} é {pii[key]}."
    return text, urgent


def _priority(rng: np.random.Generator, category: str, urgent: bool, tier: str) -> str:
    score = {"billing": 1.2, "account": 1.1, "shipping": 1.0, "technical": 0.9, "returns": 0.6}[category]
    score += 1.6 if urgent else 0.0
    score += {"gold": 0.6, "silver": 0.3, "standard": 0.0}[tier]
    score += rng.normal(0, 0.35)
    if score > 2.6:
        return "urgent"
    if score > 1.6:
        return "high"
    if score > 0.9:
        return "medium"
    return "low"


def generate(seed: int = 42, n_customers: int = 1500, n_orders: int = 6000,
             days: int = 365, start: date = date(2025, 9, 1)) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    customers = _customers(rng, n_customers, start)
    orders = _orders(rng, customers, n_orders, start, days)

    # Daily volume: base + weekly seasonality (Mondays heavier, Sundays lighter) + incidents.
    incident_days = sorted(rng.choice(np.arange(20, days), size=10, replace=False).tolist())
    incident_rows = []
    incident_map: dict[int, tuple[str, float]] = {}
    for d in incident_days:
        cat, mult, label = INCIDENTS[int(rng.integers(0, len(INCIDENTS)))]
        incident_map[int(d)] = (cat, mult)
        incident_rows.append({"date": start + timedelta(days=int(d)), "category": cat,
                              "description": label})

    base_mix = np.array([0.24, 0.30, 0.16, 0.14, 0.16])
    weekday_factor = [1.25, 1.1, 1.0, 1.0, 0.95, 0.8, 0.65]
    tickets, tid = [], 1
    orders_by_customer = orders.groupby("customer_id")
    customers_idx = customers.set_index("customer_id")

    for d in range(days):
        day = start + timedelta(days=d)
        n = rng.poisson(28 * weekday_factor[day.weekday()])
        mix = base_mix.copy()
        if d in incident_map:
            cat, mult = incident_map[d]
            extra = rng.poisson(28 * (mult - 1) * base_mix[CATEGORIES.index(cat)])
            n += extra
            mix[CATEGORIES.index(cat)] *= mult
            mix /= mix.sum()
        for _ in range(int(n)):
            category = str(rng.choice(CATEGORIES, p=mix))
            cid = int(rng.integers(1, n_customers + 1))
            cust = customers_idx.loc[cid]
            if cid in orders_by_customer.groups and rng.random() < 0.85:
                order = orders_by_customer.get_group(cid).sample(1, random_state=int(rng.integers(1e9))).iloc[0]
                order_id, carrier = int(order["order_id"]), str(order["carrier"])
            else:
                order_id, carrier = int(rng.integers(10001, 10000 + n_orders)), str(rng.choice(CARRIERS))
            pii = {"email": cust["email"], "cpf": cust["cpf"], "phone": cust["phone"]}
            body, urgent = _ticket_text(rng, category, order_id, carrier, cust["state"], pii)
            priority = _priority(rng, category, urgent, cust["tier"])
            channel = str(rng.choice(CHANNELS, p=[0.35, 0.3, 0.25, 0.1]))
            label = category
            if rng.random() < 0.05:  # annotation noise, as in real labelled data
                label = str(rng.choice(CATEGORIES))
            created = datetime.combine(day, datetime.min.time()) + timedelta(
                minutes=int(rng.integers(0, 1440)))
            resolved = rng.random() < 0.92
            hours = float(rng.gamma(2.0, RESOLUTION_MEAN_H[priority] / 2.0))
            breached = hours > SLA_TARGET_H[priority]
            csat_mean = 4.3 - 0.5 * urgent - 1.1 * breached + (0.2 if channel == "chat" else 0.0)
            tickets.append({
                "ticket_id": tid,
                "customer_id": cid,
                "order_id": order_id,
                "created_at": created,
                "channel": channel,
                "subject": body.split(",")[0][:60],
                "body": body,
                "category": label,
                "priority": priority,
                "status": "resolved" if resolved else rng.choice(["open", "pending"]),
                "resolution_hours": round(hours, 1) if resolved else None,
                "csat": int(np.clip(round(rng.normal(csat_mean, 0.8)), 1, 5)) if resolved else None,
            })
            tid += 1

    return SyntheticDataset(customers, orders, pd.DataFrame(tickets), pd.DataFrame(incident_rows))
