"""Build examples/golden.json: the reviewed golden dataset of the sample corpus.

Each question names its file and the exact evidence excerpt. The script parses the
file exactly as ingestion does, finds the excerpt in the canonical text and writes
the portable export format (content hash + offsets + excerpt), all approved.

    uv run python examples/build_golden.py
"""

import hashlib
import json
from pathlib import Path

from rag_lab.adapters.outbound.parsers.registry import ParserRegistry
from rag_lab.domain.evaluation import locate_excerpt

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"

# (question, file, evidence excerpt). Some questions reuse the document's words
# (lexical search should shine), others paraphrase them (vectors should shine).
QUESTIONS: list[tuple[str, str, str]] = [
    # returns-and-refunds.md
    ("How many days do I have to return an item?", "returns-and-refunds.md", "Most items can be returned within 45 days of delivery."),
    ("Do top-tier loyalty members get more time to send things back?", "returns-and-refunds.md", "Members of the Trailmark loyalty program at the Summit tier get an extended window of 90 days."),
    ("Can I return something with a red Final Sale tag?", "returns-and-refunds.md", 'Items bought during the winter clearance event, marked with a red "Final Sale" tag, cannot be returned at all.'),
    ("Can I bring back hiking shoes I already wore outside?", "returns-and-refunds.md", "Footwear that has been worn outdoors can still be returned if it shows no more wear than a short walk would cause."),
    ("What has to come back with a tent I am returning?", "returns-and-refunds.md", "Tents must be returned with every pole, stake and guy line that came in the box."),
    ("Why will the shop refuse a camping stove that still has gas in it?", "returns-and-refunds.md", "we cannot accept any item that still contains gas or liquid fuel, for safety reasons"),
    ("What does an RMA number look like?", "returns-and-refunds.md", "the number always begins with the letters RMA followed by eight digits, for example RMA-20481937"),
    ("What happens to a parcel sent back without the authorization number?", "returns-and-refunds.md", "Parcels that arrive without this number are refused by the warehouse and sent back to the sender."),
    ("How long until a card refund shows up on my statement?", "returns-and-refunds.md", "Card refunds appear on the statement within 5 to 7 business days after the warehouse inspects the parcel."),
    ("I paid with a gift card. How am I refunded?", "returns-and-refunds.md", "Gift card purchases are refunded as a new digital gift card, issued the same day."),
    ("Will you give me back the delivery cost if the product arrived broken?", "returns-and-refunds.md", "The original shipping fee is not refunded, except when the product was defective or we sent the wrong item."),
    ("How long is the replacement size held during an exchange?", "returns-and-refunds.md", "We reserve the replacement for 10 days while the original item travels back to us."),
    # shipping-and-delivery.md
    ("How much does standard delivery cost?", "shipping-and-delivery.md", "Standard delivery takes three to five business days and costs 6.90 per order."),
    ("From what order value is shipping free?", "shipping-and-delivery.md", "Orders above 120 ship for free with standard delivery."),
    ("What is the cut-off time to get my parcel tomorrow?", "shipping-and-delivery.md", "Express delivery arrives on the next business day for orders placed before 13:00 local time"),
    ("How long does a shop keep an order I chose to collect myself?", "shipping-and-delivery.md", "the shop keeps it for 7 days before cancelling it"),
    ("Do deliveries to the islands take longer?", "shipping-and-delivery.md", "island deliveries take two extra days"),
    ("Who pays import taxes on orders sent abroad?", "shipping-and-delivery.md", "Duties and import taxes on international orders are paid by the customer when the parcel is delivered."),
    ("Can gas canisters be sent by plane?", "shipping-and-delivery.md", "We do not ship stove fuel canisters by air."),
    ("How is a kayak delivered and how much does it cost?", "shipping-and-delivery.md", "Freight delivery takes 7 to 12 business days and costs a flat 39."),
    ("My parcel says delivered but I can't find it. What should I do?", "shipping-and-delivery.md", "customers should wait 24 hours, check with neighbours, and then contact support"),
    ("How soon must I report a box that arrived crushed?", "shipping-and-delivery.md", "Parcels damaged in transit must be reported within 48 hours of delivery"),
    # warranty-and-repairs.md
    ("How long is the legal warranty?", "warranty-and-repairs.md", "All products sold by Kestrel & Fern carry a two-year legal warranty against manufacturing defects."),
    ("Is there a lifetime guarantee on Fernline zippers?", "warranty-and-repairs.md", "Our own brand, Fernline, adds a lifetime guarantee on stitching and zippers for as long as the original buyer owns the product."),
    ("Is a jacket whose waterproofing wore off after years covered?", "warranty-and-repairs.md", "Faded colours and worn-out waterproof coatings after years of use are considered normal wear."),
    ("Who decides whether a defect is covered and how long does it take?", "warranty-and-repairs.md", "The repair workshop in Lisbon decides within 10 business days whether the defect is covered."),
    ("When do I get my money back instead of a fix under warranty?", "warranty-and-repairs.md", "A refund is only offered when neither a repair nor a replacement is possible."),
    ("How much does it cost to resole hiking boots?", "warranty-and-repairs.md", "Resoling hiking boots: 55 per pair"),
    ("What is the price of a new tent pole section?", "warranty-and-repairs.md", "Replacing a broken tent pole section: 12"),
    ("When does the workshop start a paid repair?", "warranty-and-repairs.md", "Customers receive a quote by e-mail and the work only starts after they accept it."),
    ("Why does fixing a down jacket take an extra week?", "warranty-and-repairs.md", "Down jackets and sleeping bags are cleaned before any repair, at no extra cost, because feathers cannot be sewn safely when they are dirty."),
    # trailmark-loyalty-program.md
    ("How many points do I get per euro?", "trailmark-loyalty-program.md", "Members earn one point for every euro spent."),
    ("Do I earn points when I buy a gift card?", "trailmark-loyalty-program.md", "Points are not earned on gift cards, shipping fees or repair services."),
    ("Why can't I use my points right after buying?", "trailmark-loyalty-program.md", "Points become available 30 days after the purchase, once the return window for most items has passed."),
    ("When do loyalty points expire?", "trailmark-loyalty-program.md", "Points expire 24 months after they were earned if the member makes no purchase in that period."),
    ("What do I need to reach the Ridge tier?", "trailmark-loyalty-program.md", "| Ridge | 500 – 1,499 | Free standard delivery on every order |"),
    ("How often is my membership level updated?", "trailmark-loyalty-program.md", "The tier is recalculated on the first day of every month based on the points earned in the previous 12 months."),
    ("How many points make a 10 voucher?", "trailmark-loyalty-program.md", "Every 200 points can be exchanged for a voucher worth 10."),
    ("Can I give my points to my partner?", "trailmark-loyalty-program.md", "they cannot be transferred to another member, except between two accounts registered at the same home address"),
    # product-care-guide.md
    ("Should I use fabric softener on a rain jacket?", "product-care-guide.md", "never with fabric softener, which blocks the pores of the membrane"),
    ("How do I bring back the water repellency of my jacket?", "product-care-guide.md", "Restore it with a spray-on proofer, applied to the clean, damp jacket."),
    ("I had to pack my tent in the rain. What now?", "product-care-guide.md", "If it must be packed in the rain, set it up again within 24 hours to dry it completely"),
    ("How do I set up the poles of the KF-RB2?", "product-care-guide.md", "Its poles should be assembled from the middle outward to avoid stress on the shock cord."),
    ("How should a sleeping bag be stored between trips?", "product-care-guide.md", "Store sleeping bags loosely in the large cotton storage sack, never compressed in the stuff sack"),
    ("Why put tennis balls in the dryer with a down bag?", "product-care-guide.md", "must be dried with two or three tennis balls to break up the clumps"),
    ("Can I dry wet boots next to the fireplace?", "product-care-guide.md", "Direct heat cracks the leather and weakens the glue between the sole and the upper."),
    # store-staff-handbook.md
    ("What time do stores open on Sundays?", "store-staff-handbook.md", "from 11:00 to 18:00 on Sundays"),
    ("Up to what amount can a sales assistant approve a return alone?", "store-staff-handbook.md", "Staff can approve returns up to a value of 300 on their own."),
    ("When must a rental climbing rope be taken out of service?", "store-staff-handbook.md", "retired after 50 rental days or immediately after any fall that loaded the rope"),
    ("What discount do employees get on Fernline gear?", "store-staff-handbook.md", "Employees receive a 30% discount on all products, and 40% on Fernline products, for personal use."),
    # faq.txt
    ("Can I split my payment into several months?", "faq.txt", "Orders above 200 can be paid in three monthly instalments without interest when paying with a credit card on our website."),
    ("Until when can I cancel an order?", "faq.txt", "Orders can be changed or cancelled within one hour of being placed."),
]


def main() -> None:
    parser = ParserRegistry()
    files = {}
    for path in sorted(CORPUS.iterdir()):
        data = path.read_bytes()
        files[path.name] = (hashlib.sha256(data).hexdigest(), parser.parse(path.name, data).text)
    questions = []
    for text, filename, excerpt in QUESTIONS:
        content_hash, canonical = files[filename]
        span = locate_excerpt(canonical, excerpt)
        if span is None:
            raise SystemExit(f"excerpt not found in {filename}: {excerpt!r}")
        questions.append(
            {
                "text": text,
                "status": "approved",
                "origin": "manual",
                "evidence": [
                    {
                        "content_hash": content_hash,
                        "filename": filename,
                        "start": span.start,
                        "end": span.end,
                        "excerpt": canonical[span.start : span.end],
                    }
                ],
            }
        )
    payload = {"format_version": 1, "collection": "sample-handbook", "questions": questions}
    target = HERE / "golden.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(questions)} questions to {target}")


if __name__ == "__main__":
    main()
