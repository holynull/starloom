
cmc_currency_map_api_doc="""
Base URL: https://pro-api.coinmarketcap.com/v1/cryptocurrency/map

CoinMarketCap ID Map

Returns a mapping of all cryptocurrencies to unique CoinMarketCap ids. Per our Best Practices we recommend utilizing CMC ID instead of cryptocurrency symbols to securely identify cryptocurrencies with our other endpoints and in your own application logic. Each cryptocurrency returned includes typical identifiers such as name, symbol, and token_address for flexible mapping to id.

By default this endpoint returns cryptocurrencies that have actively tracked markets on supported exchanges. You may receive a map of all inactive cryptocurrencies by passing listing_status=inactive. You may also receive a map of registered cryptocurrency projects that are listed but do not yet meet methodology requirements to have tracked markets via listing_status=untracked. Please review our methodology documentation for additional details on listing states.

Cryptocurrencies returned include first_historical_data and last_historical_data timestamps to conveniently reference historical date ranges available to query with historical time-series data endpoints. You may also use the aux parameter to only include properties you require to slim down the payload if calling this endpoint frequently.

This endpoint is available on the following API plans:
	Basic
	Hobbyist
	Startup
	Standard
	Professional
	Enterprise

Cache / Update frequency: Mapping data is updated only as needed, every 30 seconds.
Plan credit use: 1 API call credit per request no matter query size.
CMC equivalent pages: No equivalent, this data is only available via API.

PARAMETERS:
symbol:
	Type: string
    Optionally pass a comma-separated list of cryptocurrency symbols to return CoinMarketCap IDs for. If this option is passed, other options will be ignored.

Responses:
200 Success
id: The unique cryptocurrency ID for this cryptocurrency.
rank: The rank of this cryptocurrency.
name: The name of this cryptocurrency.
symbol: The ticker symbol for this cryptocurrency, always in all caps.
slug: The web URL friendly shorthand version of this cryptocurrency name.
platform: Metadata about the parent cryptocurrency platform this cryptocurrency belongs to if it is a token, otherwise null.
RESPONSE SCHEMA Sample
{{
	{
		"data": [
			{
				"id": 1,
				"rank": 1,
				"name": "Bitcoin",
				"symbol": "BTC",
				"slug": "bitcoin",
				"is_active": 1,
				"first_historical_data": "2013-04-28T18:47:21.000Z",
				"last_historical_data": "2020-05-05T20:44:01.000Z",
				"platform": null
			}
              ],
		"status": {
		"timestamp": "2018-06-02T22:51:28.209Z",
		"error_code": 0,
		"error_message": "",
		"elapsed": 10,
		"credit_count": 1
		}
	}
}}
400 Bad request
RESPONSE SCHEMA
{{
	{
		"status": {
		"timestamp": "2018-06-02T22:51:28.209Z",
		"error_code": 400,
		"error_message": "Invalid value for \"id\"",
		"elapsed": 10,
		"credit_count": 0
		}
	}
}}
    
"""

cmc_quote_lastest_api_doc="""
Base URL: https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest

Quotes Latest v2 API Documentation
Returns the latest market quote for 1 or more cryptocurrencies. Use the "convert" option to return market values in multiple fiat and cryptocurrency conversions in the same call.
There is no need to use aux to specify a specific market data, and the returned quote contains all market data.

PARAMETERS:
slug: Alternatively pass a comma-separated list of cryptocurrency slugs. Example: "bitcoin,ethereum"
symbol: Alternatively pass one or more comma-separated cryptocurrency symbols. Example: "BTC,ETH". At least one "id" or "slug" or "symbol" is required for this request.
convert: Optionally calculate market quotes in up to 120 currencies at once by passing a comma-separated list of cryptocurrency or fiat currency symbols. Each additional convert option beyond the first requires an additional call credit. A list of supported fiat options can be found here. Each conversion is returned in its own "quote" object.

RESPONSE
id: The unique CoinMarketCap ID for this cryptocurrency.
name: The name of this cryptocurrency.
symbol: The ticker symbol for this cryptocurrency.
slug: The web URL friendly shorthand version of this cryptocurrency name.
cmc_rank: The cryptocurrency's CoinMarketCap rank by market cap.
num_market_pairs: The number of active trading pairs available for this cryptocurrency across supported exchanges.
circulating_supply: The approximate number of coins circulating for this cryptocurrency.
total_supply: The approximate total amount of coins in existence right now (minus any coins that have been verifiably burned).
market_cap_by_total_supply: The market cap by total supply. This field is only returned if requested through the aux request parameter.
max_supply: The expected maximum limit of coins ever to be available for this cryptocurrency.
date_added: Timestamp (ISO 8601) of when this cryptocurrency was added to CoinMarketCap.
tags: Array of tags associated with this cryptocurrency. Currently only a mineable tag will be returned if the cryptocurrency is mineable. Additional tags will be returned in the future.
platform: Metadata about the parent cryptocurrency platform this cryptocurrency belongs to if it is a token, otherwise null.
self_reported_circulating_supply: The self reported number of coins circulating for this cryptocurrency.
self_reported_market_cap: The self reported market cap for this cryptocurrency.
quote: A map of market quotes in different currency conversions. The default map included is USD. See the flow Quote Map Instructions.

Quote Map Instructions:
price: Price in the specified currency.
volume_24h: Rolling 24 hour adjusted volume in the specified currency.
volume_change_24h: 24 hour change in the specified currencies volume.
volume_24h_reported: Rolling 24 hour reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_7d: Rolling 7 day adjusted volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_7d_reported: Rolling 7 day reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_30d: Rolling 30 day adjusted volume in the specified currency. This field is only returned if requested through the aux request parameter.
volume_30d_reported: Rolling 30 day reported volume in the specified currency. This field is only returned if requested through the aux request parameter.
market_cap: Market cap in the specified currency.
market_cap_dominance: Market cap dominance in the specified currency.
fully_diluted_market_cap: Fully diluted market cap in the specified currency.
percent_change_1h: Percentage price increase within 1 hour in the specified currency.
percent_change_24h: Percentage price increase within 24 hour in the specified currency.
percent_change_7d: Percentage price increase within 7 day in the specified currency.
percent_change_30d: Percentage price increase within 30 day in the specified currency.
"""

quotes_chain_template="""
Please turn the user input into a fully formed question.
User input: {user_input}
"""

consider_what_is_the_product="""
Question: {original_question}
The question is about the latest market trend for a certain product. Only tell me what is the product's name in the question and end with space? 
"""

api_question_template="""
What is the latest market trend of {product}?
"""

quotes_chain_answer="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

cmc_quote_historical_api_doc="""
## Instroduction
Base URL: https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/historical  
Returns an interval of historic market quotes for any cryptocurrency based on time and interval parameters.

## Technical Notes
- A historic quote for every "interval" period between your "time_start" and "time_end" will be returned.
- If a "time_start" is not supplied, the "interval" will be applied in reverse from "time_end".
- If "time_end" is not supplied, it defaults to the current time.
- At each "interval" period, the historic quote that is closest in time to the requested time will be returned.
- If no historic quotes are available in a given "interval" period up until the next interval period, it will be skipped.

## Implementation Tips
- Want to get the last quote of each UTC day? Don't use "interval=daily" as that returns the first quote. Instead use "interval=24h" to repeat a specific timestamp search every 24 hours and pass ex. "time_start=2019-01-04T23:59:00.000Z" to query for the last record of each UTC day.
- This endpoint supports requesting multiple cryptocurrencies in the same call. Please note the API response will be wrapped in an additional object in this case.

## Interval Options
There are 2 types of time interval formats that may be used for "interval".

The first are calendar year and time constants in UTC time:
"hourly" - Get the first quote available at the beginning of each calendar hour.
"daily" - Get the first quote available at the beginning of each calendar day.
"weekly" - Get the first quote available at the beginning of each calendar week.
"monthly" - Get the first quote available at the beginning of each calendar month.
"yearly" - Get the first quote available at the beginning of each calendar year.

The second are relative time intervals.
"m": Get the first quote available every "m" minutes (60 second intervals). Supported minutes are: "5m", "10m", "15m", "30m", "45m".
"h": Get the first quote available every "h" hours (3600 second intervals). Supported hour intervals are: "1h", "2h", "3h", "4h", "6h", "12h".
"d": Get the first quote available every "d" days (86400 second intervals). Supported day intervals are: "1d", "2d", "3d", "7d", "14d", "15d", "30d", "60d", "90d", "365d".

## Parameters
id: One or more comma-separated CoinMarketCap cryptocurrency IDs. Example: "1,2"
symbol: Alternatively pass one or more comma-separated cryptocurrency symbols. Example: "BTC,ETH". At least one "id" or "symbol" is required for this request.
time_start: Timestamp (Unix or ISO 8601) to start returning quotes for. Optional, if not passed, we'll return quotes calculated in reverse from "time_end".
time_end: Timestamp (Unix or ISO 8601) to stop returning quotes for (inclusive). Optional, if not passed, we'll default to the current time. If no "time_start" is passed, we return quotes in reverse order starting from this time.
count: Default 10. The number of interval periods to return results for. Optional, required if both "time_start" and "time_end" aren't supplied. The default is 10 items. The current query limit is 10000.
interval: Default "5m". Interval of time to return data points for. See details in endpoint description. Valid values: "yearly" "monthly" "weekly" "daily" "hourly" "5m" "10m" "15m""30m""45m""1h""2h""3h""4h""6h""12h""24h""1d""2d""3d""7d""14d""15d""30d""60d""90d""365d"
convert: By default market quotes are returned in USD. Optionally calculate market quotes in up to 3 other fiat currencies or cryptocurrencies.

## Response
id: The CoinMarketCap cryptocurrency ID.
name: The cryptocurrency name.
symbol: The cryptocurrency symbol.
quotes: An array of quotes for each interval for this cryptocurrency.

### Item of quotes
timestamp: Timestamp of when this historical quote was recorded.
quote: A map of market details for this quote in different currency conversions. The default map included is USD.

### Quote map
price: Aggregate 24 hour adjusted volume for all market pairs tracked for this cryptocurrency at the current historical interval.
volume_24hr: Aggregate 24 hour adjusted volume for all market pairs tracked for this cryptocurrency at the current historical interval.
market_cap: Number of market pairs available at the current historical interval.
timestamp: Timestamp (ISO 8601) of when the conversion currency's current value was referenced for this conversion.
"""