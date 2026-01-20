from core.alpaca_trader import AlpacaTrader

def test_alpaca():
    trader = AlpacaTrader()
    account = trader.get_account()
    if account:
        print(f"Success! Account Status: {account.status}")
        print(f"Equity: {account.equity}")
    else:
        print("Failed to get account. Check your API keys.")

if __name__ == "__main__":
    test_alpaca()
