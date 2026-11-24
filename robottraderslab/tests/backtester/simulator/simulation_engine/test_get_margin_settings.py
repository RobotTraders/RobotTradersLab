from robottraderslab.exchanges import MarginMode, MarginSettings


class TestGetMarginSettings:
    async def test_returns_default_leverage_for_unset_symbol(self, sim, btc_usdt_perp):
        settings = await sim.exchange.get_margin_settings([btc_usdt_perp])

        assert settings[btc_usdt_perp] == MarginSettings(
            leverage=1.0, margin_mode=MarginMode.ISOLATED
        )

    async def test_reflects_leverage_after_set(self, sim, btc_usdt_perp):
        await sim.exchange.set_leverage(btc_usdt_perp, 7.0)

        settings = await sim.exchange.get_margin_settings([btc_usdt_perp])

        assert settings[btc_usdt_perp].leverage == 7.0
