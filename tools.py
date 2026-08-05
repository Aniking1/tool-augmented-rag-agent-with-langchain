from langchain.tools import tool


@tool
def get_flight_booking(origin: str, destination: str) -> dict:
    """
    Return flight booking details between two cities.
    """

    return {
        "origin": origin,
        "destination": destination,
        "trip_type": "Round Trip",
        "departure_duration_hours": 5.5,
        "return_duration_hours": 5.5,
        "total_flight_duration_hours": 11.0,
        "ticket_price_usd": 720.0,
        "currency": "USD",
    }


@tool
def get_hotel_booking(city: str, nights: int) -> dict:
    """
    Return hotel booking details.
    """

    price_per_night = 120.0

    return {
        "city": city,
        "nights": nights,
        "price_per_night_usd": price_per_night,
        "hotel_cost_usd": nights * price_per_night,
        "currency": "USD",
    }


@tool
def convert_currency(
    from_currency: str,
    to_currency: str,
    amount: float,
) -> dict:
    """
    Convert between supported currencies.
    """

    exchange_rates = {
        ("USD", "NGN"): 1600,
        ("NGN", "USD"): 1 / 1600,
        ("USD", "GBP"): 0.75,
        ("GBP", "USD"): 1.33,
    }

    key = (from_currency.upper(), to_currency.upper())

    if key not in exchange_rates:
        return {
            "error": (
                f"Conversion from {from_currency} "
                f"to {to_currency} is not supported."
            )
        }

    rate = exchange_rates[key]

    return {
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "amount": amount,
        "converted_amount": round(amount * rate, 2),
        "exchange_rate": rate,
    }