from tools import (
    get_flight_booking,
    get_hotel_booking,
    convert_currency,
)

from rag_tool import query_internal_knowledge


print("\nFlight Tool")
print("=" * 50)
print(get_flight_booking.invoke({
    "origin": "Lagos",
    "destination": "Nairobi",
}))

print("\nHotel Tool")
print("=" * 50)
print(get_hotel_booking.invoke({
    "city": "Nairobi",
    "nights": 3,
}))

print("\nCurrency Tool")
print("=" * 50)
print(convert_currency.invoke({
    "from_currency": "USD",
    "to_currency": "NGN",
    "amount": 720,
}))

print("\nRAG Tool")
print("=" * 50)
print(query_internal_knowledge.invoke({
    "query": "conference allowance"
}))