"""Generate intent_guidelines.json from Banking77 training data.

Run: uv run python data/guidelines/build_guidelines.py
"""

import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"

# ---------------------------------------------------------------------------
# Intent metadata: definition + edge_cases
# Keys must match the category strings in the CSV exactly.
# ---------------------------------------------------------------------------
INTENT_META: dict[str, dict[str, str]] = {
    "Refund_not_showing_up": {
        "definition": "A previously approved or processed refund has not appeared in the customer's account balance or statement within the expected timeframe.",
        "edge_cases": "Distinguish from request_refund (customer hasn't asked yet) and reverted_card_payment (merchant reversal, not a refund). Key signal: customer already initiated or was promised a refund and is chasing it.",
    },
    "activate_my_card": {
        "definition": "The customer wants to activate a newly received physical or virtual card so it can be used for transactions.",
        "edge_cases": "Differs from card_not_working (card was working before) and getting_virtual_card (obtaining a card vs activating one already received).",
    },
    "age_limit": {
        "definition": "The customer is asking about minimum or maximum age requirements for opening an account, using specific services, or making certain transactions.",
        "edge_cases": "Not a general account eligibility query. Focus is specifically on age as the limiting factor.",
    },
    "apple_pay_or_google_pay": {
        "definition": "The customer wants to set up, use, or troubleshoot Apple Pay or Google Pay mobile contactless payment with their card.",
        "edge_cases": "Differs from contactless_not_working (NFC hardware on card) and card_linking (linking to other apps). Specific to Apple/Google wallet setup.",
    },
    "atm_support": {
        "definition": "The customer has a question or issue related to ATM usage, such as supported ATM networks, fees at ATMs, or general ATM functionality.",
        "edge_cases": "Use cash_withdrawal_charge if about a specific fee charged, declined_cash_withdrawal if the ATM rejected the card, or card_swallowed if the ATM kept the card.",
    },
    "automatic_top_up": {
        "definition": "The customer wants to set up, modify, or inquire about an automatic top-up rule that adds funds to their account when the balance falls below a threshold.",
        "edge_cases": "Differs from topping_up_by_card (manual one-time top-up) and top_up_limits (asking about caps). Specific to automated/recurring top-up rules.",
    },
    "balance_not_updated_after_bank_transfer": {
        "definition": "The customer's account balance has not reflected an expected update following an inbound or outbound bank transfer.",
        "edge_cases": "Use pending_transfer if the transfer is shown as pending. This intent applies when the transfer appears settled but the balance is wrong, or the update is simply delayed.",
    },
    "balance_not_updated_after_cheque_or_cash_deposit": {
        "definition": "The customer's account balance has not updated after depositing a cheque or cash.",
        "edge_cases": "Distinguish from balance_not_updated_after_bank_transfer (electronic transfer vs physical deposit) and pending_top_up (visible pending status).",
    },
    "beneficiary_not_allowed": {
        "definition": "The customer is unable to add a specific recipient or is being blocked from sending money to a particular person or account.",
        "edge_cases": "Not a general transfer failure — the block is specific to the recipient. Differs from declined_transfer (payment attempt rejected at execution) and transfer_not_received_by_recipient (sent but not arrived).",
    },
    "cancel_transfer": {
        "definition": "The customer wants to cancel or reverse a transfer that has already been initiated but may not yet have completed.",
        "edge_cases": "Key signal is intent to stop a transfer they regret initiating. Differs from failed_transfer (involuntary failure) and declined_transfer (rejected before processing).",
    },
    "card_about_to_expire": {
        "definition": "The customer's card is nearing its expiry date and they are asking about renewal, replacement, or what happens when it expires.",
        "edge_cases": "Differs from card_not_working (card already failed) and get_physical_card (requesting a new card unrelated to expiry).",
    },
    "card_acceptance": {
        "definition": "The customer is asking which merchants, countries, or networks accept their card, or why a specific merchant refused to accept it.",
        "edge_cases": "Differs from declined_card_payment (a specific transaction was declined) and country_support (asking about geographic availability of the service).",
    },
    "card_arrival": {
        "definition": "The customer is inquiring about the delivery status of a physical card that has been ordered but not yet received.",
        "edge_cases": "Differs from card_delivery_estimate (asking how long it takes in general) and order_physical_card (placing the order). Here the card is already ordered and the customer is tracking it.",
    },
    "card_delivery_estimate": {
        "definition": "The customer wants to know how many days it will take to receive a card after ordering, or what the expected delivery timeframe is.",
        "edge_cases": "Differs from card_arrival (tracking a specific in-transit card). Here the customer wants the standard delivery time estimate, often before or just after ordering.",
    },
    "card_linking": {
        "definition": "The customer wants to link their card to a third-party service, app, or external bank account.",
        "edge_cases": "Differs from apple_pay_or_google_pay (specific mobile wallets) and card_not_working (general malfunction). Focus is on the linking/pairing process.",
    },
    "card_not_working": {
        "definition": "The customer's physical or virtual card is not functioning for payments or other uses, with no more specific cause identified.",
        "edge_cases": "Use more specific intents when possible: contactless_not_working, declined_card_payment, virtual_card_not_working. This is the catch-all when the failure mode is unclear.",
    },
    "card_payment_fee_charged": {
        "definition": "The customer was charged an unexpected or unwanted fee on a card payment and wants an explanation or a refund of that fee.",
        "edge_cases": "Differs from extra_charge_on_statement (unknown charge of any type) and exchange_charge (fee specific to currency exchange).",
    },
    "card_payment_not_recognised": {
        "definition": "The customer sees a card transaction on their statement that they do not recognise and did not authorise.",
        "edge_cases": "Differs from transaction_charged_twice (recognises it but charged twice) and cash_withdrawal_not_recognised (cash, not card payment). Fraud or merchant name confusion are common triggers.",
    },
    "card_payment_wrong_exchange_rate": {
        "definition": "The customer believes an incorrect or unfavourable exchange rate was applied to a card payment made in a foreign currency.",
        "edge_cases": "Differs from exchange_rate (asking what the rate is) and exchange_charge (asking about fees). The customer has already been charged and disputes the rate applied.",
    },
    "card_swallowed": {
        "definition": "An ATM retained (swallowed or captured) the customer's physical card during a transaction and did not return it.",
        "edge_cases": "Very specific to ATM card retention. Differs from lost_or_stolen_card (card is gone but not ATM-retained) and declined_cash_withdrawal (declined but card returned).",
    },
    "cash_withdrawal_charge": {
        "definition": "The customer was charged a fee for withdrawing cash from an ATM and wants to understand or dispute that charge.",
        "edge_cases": "Differs from atm_support (general ATM questions) and cash_withdrawal_not_recognised (unrecognised transaction). The withdrawal is recognised but the fee is the issue.",
    },
    "cash_withdrawal_not_recognised": {
        "definition": "The customer sees a cash withdrawal on their account that they did not make and do not recognise.",
        "edge_cases": "Differs from card_payment_not_recognised (card purchase, not cash). Strong fraud signal. Also distinguish from wrong_amount_of_cash_received (received wrong amount but recognises the withdrawal).",
    },
    "change_pin": {
        "definition": "The customer wants to change the PIN number associated with their card.",
        "edge_cases": "Differs from pin_blocked (PIN is locked after failed attempts) and passcode_forgotten (app passcode, not card PIN).",
    },
    "compromised_card": {
        "definition": "The customer suspects their card details have been stolen, leaked, or used fraudulently without their physical card being lost.",
        "edge_cases": "Differs from lost_or_stolen_card (physical card is missing). Here the card may still be in the customer's possession but the details are believed compromised.",
    },
    "contactless_not_working": {
        "definition": "The customer's card fails for contactless (tap-to-pay / NFC) payments despite the card being otherwise functional.",
        "edge_cases": "Differs from card_not_working (general failure) and apple_pay_or_google_pay (mobile wallet NFC). The card's chip/swipe works but tap does not.",
    },
    "country_support": {
        "definition": "The customer is asking whether the account or card can be used in a specific country, or whether their country of residence is supported for registration.",
        "edge_cases": "Differs from card_acceptance (specific merchant or network) and fiat_currency_support (which currencies, not countries).",
    },
    "declined_card_payment": {
        "definition": "A card payment at a merchant or online checkout was declined or rejected.",
        "edge_cases": "Use more specific intents when the reason is known: contactless_not_working, card_acceptance, compromised_card. This is for unexplained decline at point of sale.",
    },
    "declined_cash_withdrawal": {
        "definition": "An attempt to withdraw cash at an ATM was declined or rejected.",
        "edge_cases": "Differs from card_swallowed (ATM kept the card) and cash_withdrawal_not_recognised (unauthorised withdrawal). Here the transaction was attempted and refused.",
    },
    "declined_transfer": {
        "definition": "A transfer initiated by the customer was declined or rejected before or during processing.",
        "edge_cases": "Differs from failed_transfer (attempted but failed mid-process) and cancel_transfer (customer wants to stop it). Here the system actively refused the transfer.",
    },
    "direct_debit_payment_not_recognised": {
        "definition": "The customer sees a direct debit charge on their account that they did not authorise or do not recognise.",
        "edge_cases": "Specific to direct debits (recurring bank-authorised debits), not one-off card transactions. Differs from card_payment_not_recognised and extra_charge_on_statement.",
    },
    "disposable_card_limits": {
        "definition": "The customer is asking about spending limits, validity period, or usage restrictions on a disposable (single-use) virtual card.",
        "edge_cases": "Differs from top_up_limits (account top-up caps) and get_disposable_virtual_card (requesting one). Focus is on understanding the limits of an already-obtained disposable card.",
    },
    "edit_personal_details": {
        "definition": "The customer wants to update personal information such as name, address, phone number, or email address on their account.",
        "edge_cases": "Not identity verification — this is about changing stored profile data. Differs from verify_my_identity (KYC process) and unable_to_verify_identity (verification failure).",
    },
    "exchange_charge": {
        "definition": "The customer is asking about fees charged when converting or exchanging currency.",
        "edge_cases": "Differs from exchange_rate (the rate itself, not the fee) and card_payment_fee_charged (generic card fee). Focus is specifically on the cost of currency conversion.",
    },
    "exchange_rate": {
        "definition": "The customer wants to know the current exchange rate or how the app determines rates for foreign currency transactions.",
        "edge_cases": "Differs from exchange_charge (fees, not rates) and card_payment_wrong_exchange_rate (disputing a rate already applied). This is an informational query about current rates.",
    },
    "exchange_via_app": {
        "definition": "The customer wants to convert or exchange currency using the in-app currency exchange feature.",
        "edge_cases": "Differs from exchange_rate (asking about rates only) and topping_up_by_card (adding money, not converting). Focus is on actively using the in-app exchange tool.",
    },
    "extra_charge_on_statement": {
        "definition": "The customer sees an unexpected or unexplained charge on their statement and wants to understand what it is.",
        "edge_cases": "More specific intents should be preferred when identifiable: card_payment_fee_charged, cash_withdrawal_charge, exchange_charge. Use this when the charge type is unknown.",
    },
    "failed_transfer": {
        "definition": "A transfer was initiated and attempted but failed to complete, with uncertainty about whether the funds left the account.",
        "edge_cases": "Differs from declined_transfer (rejected before processing) and cancel_transfer (customer-initiated stop). Here the transfer started but something went wrong mid-process.",
    },
    "fiat_currency_support": {
        "definition": "The customer is asking which traditional (fiat) currencies are supported for holding balances, sending, or receiving money.",
        "edge_cases": "Differs from country_support (countries, not currencies) and supported_cards_and_currencies (broader question also including card networks).",
    },
    "get_disposable_virtual_card": {
        "definition": "The customer wants to obtain a new single-use or disposable virtual card number for secure online purchases.",
        "edge_cases": "Differs from getting_virtual_card (permanent virtual card) and disposable_card_limits (asking about limits of one already held).",
    },
    "get_physical_card": {
        "definition": "The customer wants to request or order a physical card to be mailed to their address.",
        "edge_cases": "Closely related to order_physical_card. Use get_physical_card when the customer is asking how to get one; order_physical_card when placing or tracking a specific order.",
    },
    "getting_spare_card": {
        "definition": "The customer wants an additional physical card for the same account, either as a backup or for a secondary user.",
        "edge_cases": "Differs from get_physical_card (first-time request) and lost_or_stolen_card (replacement after loss). Here the customer already has a card and wants an extra one.",
    },
    "getting_virtual_card": {
        "definition": "The customer wants to get or access a permanent virtual card number for online payments.",
        "edge_cases": "Differs from get_disposable_virtual_card (one-time use) and virtual_card_not_working (already has one but it's broken). Focus is on obtaining a reusable virtual card.",
    },
    "lost_or_stolen_card": {
        "definition": "The customer's physical card has been lost or stolen and they need to report it, freeze it, or request a replacement.",
        "edge_cases": "Differs from compromised_card (card in hand but details stolen) and card_not_working (card present but malfunctioning). Physical card is missing.",
    },
    "lost_or_stolen_phone": {
        "definition": "The customer's phone has been lost or stolen, impacting their access to the banking app and potentially their financial security.",
        "edge_cases": "Differs from lost_or_stolen_card (card, not phone). The concern is app access, linked card security, or apple/google pay exposure on the missing device.",
    },
    "order_physical_card": {
        "definition": "The customer is placing or tracking an order for a new physical card.",
        "edge_cases": "Closely related to get_physical_card. Use order_physical_card when the customer explicitly mentions ordering or has already ordered. Overlaps with card_arrival for in-transit cards.",
    },
    "passcode_forgotten": {
        "definition": "The customer has forgotten their app login passcode or PIN and needs to reset or recover access.",
        "edge_cases": "Differs from pin_blocked (card PIN locked, not app passcode) and change_pin (wants to change, not forgotten). Focus is on app access recovery.",
    },
    "pending_card_payment": {
        "definition": "A card payment is showing as pending rather than settled, and the customer has questions about when it will complete or why it is stuck.",
        "edge_cases": "Differs from declined_card_payment (rejected) and card_payment_not_recognised (unknown charge). The transaction is recognised but not yet settled.",
    },
    "pending_cash_withdrawal": {
        "definition": "A cash withdrawal is showing as pending on the account rather than having fully settled.",
        "edge_cases": "Differs from declined_cash_withdrawal (rejected) and cash_withdrawal_not_recognised (unknown). The withdrawal is recognised but pending settlement.",
    },
    "pending_top_up": {
        "definition": "A top-up (account deposit) is showing as pending and the funds have not yet been credited to the account.",
        "edge_cases": "Differs from top_up_failed (failed, not pending) and balance_not_updated_after_bank_transfer (no pending indicator visible). The top-up is visible as pending.",
    },
    "pending_transfer": {
        "definition": "A transfer is showing as pending and the recipient has not yet received the funds.",
        "edge_cases": "Differs from failed_transfer (error occurred) and transfer_not_received_by_recipient (sender believes it completed). Here the transfer is clearly in a pending state.",
    },
    "pin_blocked": {
        "definition": "The customer's card PIN has been blocked, typically after too many consecutive incorrect PIN entry attempts.",
        "edge_cases": "Differs from change_pin (wants a new PIN) and passcode_forgotten (app passcode). Focus is on the blocked state requiring unblock, not just resetting.",
    },
    "receiving_money": {
        "definition": "The customer wants to know how to receive money into their account, or has a general question about incoming transfers.",
        "edge_cases": "Differs from transfer_into_account (customer is doing the sending) and transfer_not_received_by_recipient (a specific transfer has gone missing). This is about how the receiving process works.",
    },
    "request_refund": {
        "definition": "The customer wants to request a refund for a transaction and believes they are entitled to get their money back.",
        "edge_cases": "Differs from Refund_not_showing_up (refund already requested and expected). Here the customer is initiating the refund request, not chasing one already in progress.",
    },
    "reverted_card_payment?": {
        "definition": "A card payment that was previously processed has been reversed or reverted, and the customer has questions about why this happened or what to expect.",
        "edge_cases": "Differs from request_refund (customer-initiated return) and Refund_not_showing_up (refund delayed). Here the reversal was initiated by the merchant or system, not the customer.",
    },
    "supported_cards_and_currencies": {
        "definition": "The customer wants to know which card networks (Visa, Mastercard) or which currencies are supported by the service.",
        "edge_cases": "Broader than visa_or_mastercard (card network only) or fiat_currency_support (currency only). Use this when the query covers both or is general.",
    },
    "terminate_account": {
        "definition": "The customer wants to permanently close or delete their account and all associated data.",
        "edge_cases": "Not a temporary freeze or suspension. Differs from lost_or_stolen_card (card issue, not account closure). Irreversible action — clear intent signal required.",
    },
    "top_up_by_bank_transfer_charge": {
        "definition": "The customer is asking about or disputing a fee that was charged for topping up their account via bank transfer.",
        "edge_cases": "Differs from top_up_by_card_charge (card top-up fee) and transfer_fee_charged (outgoing transfer fee). Focus is specifically on the bank transfer top-up method fee.",
    },
    "top_up_by_card_charge": {
        "definition": "The customer is asking about or disputing a fee charged for topping up their account using a debit or credit card.",
        "edge_cases": "Differs from top_up_by_bank_transfer_charge (bank transfer method) and topping_up_by_card (the act of topping up, not the fee).",
    },
    "top_up_by_cash_or_cheque": {
        "definition": "The customer wants to know how to add money to their account using physical cash or a cheque.",
        "edge_cases": "Differs from topping_up_by_card (card method) and top_up_by_bank_transfer_charge (bank transfer method). Specific to cash or cheque deposit method.",
    },
    "top_up_failed": {
        "definition": "An attempt to add funds to the account failed and the money was not credited.",
        "edge_cases": "Differs from pending_top_up (visible as pending, not failed) and top_up_reverted (succeeded then reversed). The top-up attempt was made and resulted in an error.",
    },
    "top_up_limits": {
        "definition": "The customer is asking about minimum or maximum limits on how much they can top up their account per transaction or per period.",
        "edge_cases": "Differs from disposable_card_limits (limits on a disposable card) and automatic_top_up (automated rules). Focus is on the cap amounts for funding the account.",
    },
    "top_up_reverted": {
        "definition": "A top-up that was previously successfully added to the account has been reversed and the funds removed.",
        "edge_cases": "Differs from top_up_failed (never succeeded) and pending_top_up (still processing). The money appeared and then was taken back.",
    },
    "topping_up_by_card": {
        "definition": "The customer wants to add money to their account using a debit or credit card and has a question about the process.",
        "edge_cases": "Differs from top_up_by_card_charge (asking about the fee, not the process) and automatic_top_up (automated rule). This is about the act of manually topping up via card.",
    },
    "transaction_charged_twice": {
        "definition": "The customer was charged twice for the same transaction, resulting in a duplicate debit.",
        "edge_cases": "Differs from card_payment_not_recognised (unknown charge) and extra_charge_on_statement (unknown extra). Here the customer recognises both charges as the same transaction duplicated.",
    },
    "transfer_fee_charged": {
        "definition": "The customer was charged a fee for sending an outbound transfer and wants an explanation or dispute of that fee.",
        "edge_cases": "Differs from top_up_by_bank_transfer_charge (incoming top-up fee) and exchange_charge (currency conversion fee). Focus is on outgoing transfer fees.",
    },
    "transfer_into_account": {
        "definition": "The customer wants to transfer money into their account from an external bank and has a question about how to do it.",
        "edge_cases": "Differs from receiving_money (general incoming money questions) and pending_top_up (a top-up already in progress). This is about the mechanics of sending to the account.",
    },
    "transfer_not_received_by_recipient": {
        "definition": "The customer sent a transfer that appeared to complete on their end, but the recipient has not received the funds.",
        "edge_cases": "Differs from pending_transfer (clearly still pending) and failed_transfer (error shown). Here the sender's account shows the money left but the recipient sees nothing.",
    },
    "transfer_timing": {
        "definition": "The customer wants to know how long a transfer will take to be processed, sent, or received by the recipient.",
        "edge_cases": "Differs from pending_transfer (a specific transfer is pending) and transfer_not_received_by_recipient (transfer seems lost). This is an informational query about expected timing.",
    },
    "unable_to_verify_identity": {
        "definition": "The customer is experiencing technical or procedural difficulty completing the required identity verification (KYC) process in the app.",
        "edge_cases": "Differs from verify_my_identity (wants to know how) and why_verify_identity (asking why it's required). The customer is actively trying and failing to verify.",
    },
    "verify_my_identity": {
        "definition": "The customer needs to complete an identity verification step required by the app and is asking how to do it.",
        "edge_cases": "Differs from unable_to_verify_identity (trying and failing) and why_verify_identity (questioning the requirement). Here the customer is willing and looking for instructions.",
    },
    "verify_source_of_funds": {
        "definition": "The customer has been asked by the app to prove where their funds come from as part of a compliance or anti-money-laundering check.",
        "edge_cases": "Distinct from verify_my_identity (personal ID check). This is specifically about documenting the origin of deposited or transferred funds.",
    },
    "verify_top_up": {
        "definition": "The customer needs to verify or confirm a specific top-up transaction, typically as part of a security or compliance check.",
        "edge_cases": "Differs from verify_source_of_funds (broader AML check) and pending_top_up (waiting for funds). Specific to confirming a top-up event.",
    },
    "virtual_card_not_working": {
        "definition": "The customer's virtual card is not functioning for online payments or in-app purchases.",
        "edge_cases": "Differs from card_not_working (physical card) and getting_virtual_card (requesting one). The virtual card exists but is malfunctioning.",
    },
    "visa_or_mastercard": {
        "definition": "The customer wants to know whether their card is issued on the Visa or Mastercard network, or is asking about differences between the two.",
        "edge_cases": "Differs from supported_cards_and_currencies (broader). Use when the query is specifically about which network a card is on.",
    },
    "why_verify_identity": {
        "definition": "The customer is questioning or challenging the reason they are being asked to verify their identity.",
        "edge_cases": "Differs from verify_my_identity (willing to verify, wants instructions) and unable_to_verify_identity (trying but failing). The customer is asking WHY, not how.",
    },
    "wrong_amount_of_cash_received": {
        "definition": "The customer received a different (usually lesser) amount of cash from an ATM than they requested.",
        "edge_cases": "Differs from cash_withdrawal_not_recognised (doesn't recognise the withdrawal at all). Here the customer made the withdrawal but the amount dispensed was wrong.",
    },
    "wrong_exchange_rate_for_cash_withdrawal": {
        "definition": "An incorrect or unexpected exchange rate was applied to a cash withdrawal made in a foreign currency.",
        "edge_cases": "Differs from card_payment_wrong_exchange_rate (card purchase, not ATM cash) and exchange_rate (asking what the rate is). Specific to ATM cash withdrawal rate disputes.",
    },
}


def fetch_csv(filename: str) -> pd.DataFrame:
    url = f"{BASE}/{filename}"
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        df = pd.read_csv(io.BytesIO(resp.read()))
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_confusion_map(intent_names: list[str], top_k: int = 3) -> dict[str, list[str]]:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    readable = [n.replace("_", " ").replace("?", "") for n in intent_names]
    embeddings: np.ndarray = model.encode(readable, normalize_embeddings=True)
    sim: np.ndarray = embeddings @ embeddings.T
    np.fill_diagonal(sim, -1.0)

    confusion: dict[str, list[str]] = {}
    for i, name in enumerate(intent_names):
        top_idx = np.argsort(sim[i])[::-1][:top_k]
        confusion[name] = [intent_names[j] for j in top_idx]
    return confusion


def main() -> None:
    print("Loading training data...")
    train = fetch_csv("train.csv")

    intent_names = sorted(train["category"].unique().tolist())
    assert len(intent_names) == 77, f"Expected 77 intents, got {len(intent_names)}"

    # Sample up to 3 real queries per intent
    examples_by_intent: dict[str, list[str]] = {}
    for intent in intent_names:
        rows = train[train["category"] == intent]["text"].tolist()
        examples_by_intent[intent] = rows[:3]

    print("Computing confusion map via embeddings...")
    confusion = compute_confusion_map(intent_names)

    print("Building guidelines...")
    guidelines = []
    missing_meta = []
    for intent in intent_names:
        meta = INTENT_META.get(intent)
        if meta is None:
            missing_meta.append(intent)
            continue
        guidelines.append(
            {
                "intent_name": intent,
                "definition": meta["definition"],
                "example_queries": examples_by_intent.get(intent, []),
                "edge_cases": meta["edge_cases"],
                "commonly_confused_with": confusion[intent],
            }
        )

    if missing_meta:
        raise ValueError(f"Missing metadata for: {missing_meta}")

    out = Path(__file__).parent / "intent_guidelines.json"
    out.write_text(json.dumps(guidelines, indent=2))
    print(f"Saved {len(guidelines)} intent guidelines → {out}")


if __name__ == "__main__":
    main()
