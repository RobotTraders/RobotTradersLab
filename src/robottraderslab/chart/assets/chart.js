(function () {
  const LWC = window.LightweightCharts;
  const payload = window.__CHART__;

  const THEME = {
    background: "#0d1117",
    text: "#c9d1d9",
    grid: "#1c2128",
    border: "#30363d",
    up: "#089981",
    down: "#f23645",
  };

  const FILL = {
    upStrong: "rgba(8, 153, 129, 0.35)",
    upFaint: "rgba(8, 153, 129, 0.02)",
    downStrong: "rgba(242, 54, 69, 0.35)",
    downFaint: "rgba(242, 54, 69, 0.18)",
    none: "rgba(0, 0, 0, 0)",
  };

  const MARKER_STYLE = {
    buy: { position: "belowBar", shape: "arrowUp", color: THEME.up },
    sell: { position: "aboveBar", shape: "arrowDown", color: THEME.down },
  };

  const CANDLE = { time: 0, open: 1, high: 2, low: 3, close: 4 };
  const POINT = { time: 0, value: 1 };
  const PANE_INDEX = { price: 0, separate: 1 };
  const SEPARATE_PANE = "separate";
  const HISTOGRAM_SHAPE = "histogram";
  const MIN_PANE_HEIGHT = 90;

  const UNDERLAY_SCALE = "underlay";
  const PERFORMANCE_KEY = "performance";
  const PRICE_TOGGLES = "price-toggles";
  const PERFORMANCE_TOGGLES = "toggles";
  const KEY_STATS = "key-stats";
  const EQUITY_LABEL = "Equity";
  const DRAWDOWN_LABEL = "Drawdown";
  const COLLAPSED_CLASS = "panel-collapsed";
  const LINE_WIDTH = 2;
  const PRICE_FORMAT = { type: "price", precision: 2, minMove: 0.01 };
  const PERCENT_FORMAT = { type: "percent", precision: 2, minMove: 0.01 };
  const DRAWDOWN_FORMAT =
    payload.drawdownUnit === "currency" ? PRICE_FORMAT : PERCENT_FORMAT;
  const REASONS_LABEL = "Reasons";
  const PROFILE_PANEL_GAP = 8;

  const MILLISECONDS_PER_SECOND = 1000;
  const ISO_MINUTE_LENGTH = "2000-01-01T00:00".length;
  const TRADE_PADDING_BARS = 20;
  const NUMERIC_CLASS = "numeric";
  const LOCATE_CLASS = "locate-cell";
  const LOCATE_ICON =
    '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">' +
    '<circle cx="8" cy="8" r="4.2" fill="none" stroke="currentColor" stroke-width="1.4"/>' +
    '<path d="M8 0.5v3M8 12.5v3M0.5 8h3M12.5 8h3" stroke="currentColor" stroke-width="1.4"/>' +
    "</svg>";

  let activeMarketIndex = 0;

  function activeMarket() {
    return payload.markets[activeMarketIndex];
  }

  function baseOptions() {
    return {
      layout: {
        background: { type: "solid", color: THEME.background },
        textColor: THEME.text,
        attributionLogo: true,
      },
      grid: {
        vertLines: { color: THEME.grid },
        horzLines: { color: THEME.grid },
      },
      rightPriceScale: { borderColor: THEME.border },
      timeScale: { borderColor: THEME.border, timeVisible: true },
      crosshair: { mode: LWC.CrosshairMode.Normal },
      autoSize: true,
    };
  }

  function expandCandles(rows) {
    return rows.map(function (row) {
      return {
        time: row[CANDLE.time],
        open: row[CANDLE.open],
        high: row[CANDLE.high],
        low: row[CANDLE.low],
        close: row[CANDLE.close],
      };
    });
  }

  function expandPoints(rows) {
    return rows.map(function (row) {
      return { time: row[POINT.time], value: row[POINT.value] };
    });
  }

  function renderMarketMenu(onSwitch) {
    const container = document.getElementById("profiles");
    if (payload.links.length > 0) {
      container.appendChild(
        buildSwitcherMenu(
          payload.links.map(function (link) {
            return {
              label: link.label,
              settings: link.settings,
              current: link.current,
              href: link.href,
            };
          })
        )
      );
      return;
    }
    if (payload.markets.length > 1) {
      container.appendChild(
        buildSwitcherMenu(
          payload.markets.map(function (market, index) {
            return {
              label: market.label,
              settings: market.settings,
              current: index === activeMarketIndex,
              onSelect: function () {
                onSwitch(index);
              },
            };
          })
        )
      );
      return;
    }
    const label = document.createElement("span");
    label.textContent = payload.markets[0].label;
    container.appendChild(label);
  }

  function buildSwitcherMenu(items) {
    const menu = document.createElement("div");
    menu.className = "profile-menu";
    const button = buildSwitcherButton(items);
    const list = document.createElement("div");
    list.className = "profile-list";
    list.setAttribute("role", "listbox");
    list.setAttribute("aria-label", "Profile");
    list.hidden = true;
    const panel = document.createElement("div");
    panel.className = "profile-settings";
    panel.hidden = true;

    items.forEach(function (item) {
      list.appendChild(buildSwitcherOption(item, menu, panel, button, list));
    });

    button.addEventListener("click", function () {
      const opening = list.hidden;
      list.hidden = !opening;
      panel.hidden = true;
      button.setAttribute("aria-expanded", String(opening));
    });
    document.addEventListener("click", function (event) {
      if (!menu.contains(event.target)) closeProfileMenu(button, list, panel);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closeProfileMenu(button, list, panel);
    });
    menu.addEventListener("mouseleave", function () {
      panel.hidden = true;
    });

    menu.appendChild(button);
    menu.appendChild(list);
    menu.appendChild(panel);
    return menu;
  }

  function buildSwitcherButton(items) {
    const current = items.find(function (item) {
      return item.current;
    });
    const button = document.createElement("button");
    button.type = "button";
    button.className = "profile-select";
    button.setAttribute("aria-haspopup", "listbox");
    button.setAttribute("aria-expanded", "false");
    button.textContent = (current || items[0]).label;
    return button;
  }

  function buildSwitcherOption(item, menu, panel, button, list) {
    const option = document.createElement(item.href ? "a" : "button");
    option.className = "profile-option";
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", String(item.current));
    if (item.href) {
      option.href = item.href;
    } else {
      option.type = "button";
      option.addEventListener("click", function () {
        item.onSelect();
        button.textContent = item.label;
        closeProfileMenu(button, list, panel);
      });
    }
    option.textContent = item.label;
    option.addEventListener("mouseenter", function () {
      showProfileSettings(panel, menu, option, item);
    });
    return option;
  }

  function showProfileSettings(panel, menu, option, item) {
    if (item.settings.length === 0) {
      panel.hidden = true;
      return;
    }
    const table = document.createElement("table");
    item.settings.forEach(function (setting) {
      const row = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = setting.name;
      const value = document.createElement("td");
      value.textContent = setting.value;
      row.appendChild(name);
      row.appendChild(value);
      table.appendChild(row);
    });
    panel.replaceChildren(table);
    const menuBox = menu.getBoundingClientRect();
    const optionBox = option.getBoundingClientRect();
    panel.style.top = optionBox.top - menuBox.top + "px";
    panel.style.left = optionBox.right - menuBox.left + PROFILE_PANEL_GAP + "px";
    panel.hidden = false;
  }

  function closeProfileMenu(button, list, panel) {
    list.hidden = true;
    panel.hidden = true;
    button.setAttribute("aria-expanded", "false");
  }

  function drawCandles(chart, market) {
    const candles = chart.addSeries(LWC.CandlestickSeries, {
      upColor: THEME.up,
      downColor: THEME.down,
      borderVisible: false,
      wickUpColor: THEME.up,
      wickDownColor: THEME.down,
    });
    candles.setData(expandCandles(market.candles));
    return candles;
  }

  function drawLineSeries(chart, series) {
    return chart.addSeries(
      LWC.LineSeries,
      {
        color: series.color,
        title: series.name,
        lineWidth: LINE_WIDTH,
        priceLineVisible: false,
      },
      PANE_INDEX[series.pane]
    );
  }

  function drawHistogramSeries(chart, series) {
    return chart.addSeries(
      LWC.HistogramSeries,
      {
        color: series.color,
        title: series.name,
        base: 0,
        priceLineVisible: false,
      },
      PANE_INDEX[series.pane]
    );
  }

  const SERIES_SHAPES = {
    line: drawLineSeries,
    histogram: drawHistogramSeries,
  };

  function barsFirst(series) {
    const bars = series.filter(function (one) {
      return one.shape === HISTOGRAM_SHAPE;
    });
    const curves = series.filter(function (one) {
      return one.shape !== HISTOGRAM_SHAPE;
    });
    return bars.concat(curves);
  }

  function drawIndicators(chart, market) {
    const ordered = barsFirst(market.series);
    const needsSeparatePane = ordered.some(function (series) {
      return series.pane === SEPARATE_PANE;
    });
    if (needsSeparatePane && chart.panes().length === 1) {
      chart.addPane();
    }

    const handles = ordered.map(function (series) {
      const added = SERIES_SHAPES[series.shape](chart, series);
      added.setData(expandPoints(series.points));
      return added;
    });

    const toggles = ordered.map(function (series, index) {
      const added = handles[index];
      return addToggle(PRICE_TOGGLES, series.name, series.color, false, function (visible) {
        added.applyOptions({ visible: visible });
      });
    });

    return {
      clear: function () {
        handles.forEach(function (handle) {
          chart.removeSeries(handle);
        });
        toggles.forEach(function (toggle) {
          toggle.remove();
        });
      },
    };
  }

  function styledMarker(marker, withLabel) {
    const style = MARKER_STYLE[marker.side];
    return {
      time: marker.time,
      position: style.position,
      shape: style.shape,
      color: style.color,
      text: withLabel ? marker.label : "",
    };
  }

  function labelledMarkers(market, withLabels) {
    return market.markers.map(function (marker) {
      return styledMarker(marker, withLabels);
    });
  }

  function drawMarkers(candles, market) {
    let reasonsVisible = true;
    const markers = LWC.createSeriesMarkers(candles, labelledMarkers(market, true));
    addToggle(PRICE_TOGGLES, REASONS_LABEL, THEME.text, false, function (visible) {
      reasonsVisible = visible;
      markers.setMarkers(labelledMarkers(market, visible));
    });
    return {
      setMarket: function (nextMarket) {
        markers.setMarkers(labelledMarkers(nextMarket, reasonsVisible));
      },
    };
  }

  function buildPriceChart() {
    const market = activeMarket();
    const chart = LWC.createChart(document.getElementById("price"), baseOptions());
    const candles = drawCandles(chart, market);
    const markers = drawMarkers(candles, market);
    let indicators = drawIndicators(chart, market);
    chart.timeScale().fitContent();
    return {
      chart: chart,
      switchMarket: function (nextMarket) {
        candles.setData(expandCandles(nextMarket.candles));
        markers.setMarket(nextMarket);
        indicators.clear();
        indicators = drawIndicators(chart, nextMarket);
        chart.timeScale().fitContent();
      },
    };
  }

  function drawEquityArea(chart) {
    const equity = chart.addSeries(LWC.AreaSeries, {
      lineColor: THEME.up,
      topColor: FILL.upStrong,
      bottomColor: FILL.upFaint,
      lineWidth: LINE_WIDTH,
      priceLineVisible: false,
      priceFormat: PRICE_FORMAT,
    });
    equity.setData(expandPoints(payload.equity));
  }

  function drawEquityLine(chart) {
    const equity = chart.addSeries(LWC.LineSeries, {
      color: THEME.up,
      lineWidth: LINE_WIDTH,
      priceLineVisible: false,
      priceFormat: PRICE_FORMAT,
    });
    equity.setData(expandPoints(payload.equity));
  }

  function drawDrawdownCurve(chart) {
    const drawdown = chart.addSeries(LWC.BaselineSeries, {
      baseValue: { type: "price", price: 0 },
      topLineColor: THEME.down,
      topFillColor1: FILL.none,
      topFillColor2: FILL.none,
      bottomLineColor: THEME.down,
      bottomFillColor1: FILL.downStrong,
      bottomFillColor2: FILL.downStrong,
      lineWidth: LINE_WIDTH,
      priceLineVisible: false,
      priceFormat: DRAWDOWN_FORMAT,
    });
    drawdown.setData(expandPoints(payload.drawdown));
  }

  function drawDrawdownUnderlay(chart) {
    const drawdown = chart.addSeries(LWC.HistogramSeries, {
      priceScaleId: UNDERLAY_SCALE,
      color: FILL.downFaint,
      priceLineVisible: false,
      priceFormat: DRAWDOWN_FORMAT,
    });
    drawdown.setData(expandPoints(payload.drawdown));
    chart
      .priceScale(UNDERLAY_SCALE)
      .applyOptions({ scaleMargins: { top: 0, bottom: 0 } });
  }

  function buildPerformanceChart(shown) {
    const chart = LWC.createChart(
      document.getElementById(PERFORMANCE_KEY),
      baseOptions()
    );

    if (shown.equity && shown.drawdown) {
      drawDrawdownUnderlay(chart);
      drawEquityLine(chart);
    } else if (shown.equity) {
      drawEquityArea(chart);
    } else {
      drawDrawdownCurve(chart);
    }

    chart.timeScale().fitContent();
    return chart;
  }


  function candleRange(market) {
    if (market.candles.length === 0) {
      return null;
    }
    const rows = market.candles;
    return {
      from: rows[0][CANDLE.time],
      to: rows[rows.length - 1][CANDLE.time],
    };
  }

  function createSynchroniser(leader) {
    let syncing = false;
    const pushedToFollower = new Map();

    function push(target) {
      return function (range) {
        if (syncing || range === null) {
          return;
        }
        syncing = true;
        target.timeScale().setVisibleRange(range);
        syncing = false;
      };
    }

    return {
      follow: function (chart) {
        const toFollower = push(chart);
        leader.timeScale().subscribeVisibleTimeRangeChange(toFollower);
        chart.timeScale().subscribeVisibleTimeRangeChange(push(leader));
        pushedToFollower.set(chart, toFollower);
      },
      release: function (chart) {
        leader
          .timeScale()
          .unsubscribeVisibleTimeRangeChange(pushedToFollower.get(chart));
        pushedToFollower.delete(chart);
      },
      align: function (chart) {
        const range = leader.timeScale().getVisibleRange() || candleRange(activeMarket());
        if (range !== null) {
          chart.timeScale().setVisibleRange(range);
        }
      },
    };
  }

  function addToggle(container, label, colour, startHidden, onChange) {
    const button = document.createElement("button");
    button.className = "toggle";
    button.type = "button";
    button.textContent = label;
    button.style.setProperty("--toggle-dot", colour);
    document.getElementById(container).appendChild(button);

    let visible = !startHidden;
    function apply() {
      button.setAttribute("aria-pressed", String(visible));
      onChange(visible);
    }
    button.addEventListener("click", function () {
      visible = !visible;
      apply();
    });
    apply();
    return button;
  }

  function mountPerformance(synchroniser) {
    const element = document.getElementById(PERFORMANCE_KEY + "-panel");
    const shown = { equity: payload.equity.length > 0, drawdown: payload.drawdown.length > 0 };
    if (!shown.equity && !shown.drawdown) {
      element.remove();
      return;
    }

    const canvas = document.getElementById(PERFORMANCE_KEY);
    let chart = null;
    let mounted = false;

    function redraw() {
      if (!mounted) {
        return;
      }
      if (chart !== null) {
        synchroniser.release(chart);
        chart.remove();
        chart = null;
      }
      const empty = !shown.equity && !shown.drawdown;
      canvas.hidden = empty;
      element.classList.toggle(COLLAPSED_CLASS, empty);
      if (empty) {
        return;
      }
      chart = buildPerformanceChart(shown);
      synchroniser.follow(chart);
      synchroniser.align(chart);
    }

    if (payload.equity.length > 0) {
      addToggle(PERFORMANCE_TOGGLES, EQUITY_LABEL, THEME.up, false, function (visible) {
        shown.equity = visible;
        redraw();
      });
    }
    if (payload.drawdown.length > 0) {
      addToggle(
        PERFORMANCE_TOGGLES,
        DRAWDOWN_LABEL,
        FILL.downStrong,
        false,
        function (visible) {
          shown.drawdown = visible;
          redraw();
        }
      );
    }
    mounted = true;
    redraw();
  }

  function formatTime(seconds) {
    if (seconds === null) {
      return "";
    }
    return new Date(seconds * MILLISECONDS_PER_SECOND)
      .toISOString()
      .slice(0, ISO_MINUTE_LENGTH)
      .replace("T", " ");
  }

  function formatPrice(value) {
    return value === null ? "" : value.toFixed(PRICE_FORMAT.precision);
  }

  function formatPercent(value) {
    return value.toFixed(PERCENT_FORMAT.precision) + "%";
  }

  const TRADE_COLUMNS = [
    {
      label: "Side",
      text: function (trade) {
        return trade.side;
      },
    },
    {
      label: "Entry time",
      text: function (trade) {
        return formatTime(trade.entry_time);
      },
      locate: function (trade) {
        return trade.entry_time;
      },
    },
    {
      label: "Exit time",
      text: function (trade) {
        return formatTime(trade.exit_time);
      },
      locate: function (trade) {
        return trade.exit_time;
      },
    },
    {
      label: "Entry price",
      text: function (trade) {
        return formatPrice(trade.entry_price);
      },
    },
    {
      label: "Exit price",
      text: function (trade) {
        return formatPrice(trade.exit_price);
      },
    },
    {
      label: "Net PnL",
      text: function (trade) {
        return formatPrice(trade.net_pnl);
      },
      sign: function (trade) {
        return trade.net_pnl;
      },
    },
    {
      label: "Net PnL %",
      text: function (trade) {
        return formatPercent(trade.net_pnl_pct);
      },
      sign: function (trade) {
        return trade.net_pnl_pct;
      },
    },
    {
      label: "Entry reason",
      text: function (trade) {
        return trade.entry_reason;
      },
    },
    {
      label: "Exit reason",
      text: function (trade) {
        return trade.exit_reason;
      },
    },
  ];

  function momentRange(market, time) {
    const candles = candleRange(market);
    const spacing =
      (candles.to - candles.from) / Math.max(market.candles.length - 1, 1);
    const padding = spacing * TRADE_PADDING_BARS;
    return { from: time - padding, to: time + padding };
  }

  function addLocateButton(cell, market, time, priceChart) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = LOCATE_CLASS;
    button.title = "Show on chart";
    button.innerHTML = LOCATE_ICON;
    button.addEventListener("click", function () {
      priceChart.timeScale().setVisibleRange(momentRange(market, time));
    });
    cell.appendChild(button);
  }

  function buildTradeHead() {
    const head = document.createElement("thead");
    const row = head.insertRow();
    TRADE_COLUMNS.forEach(function (column) {
      const cell = document.createElement("th");
      cell.textContent = column.label;
      row.appendChild(cell);
    });
    return head;
  }

  function buildTradeRow(body, trade, market, priceChart) {
    const row = body.insertRow();
    TRADE_COLUMNS.forEach(function (column) {
      const cell = row.insertCell();
      if (column.sign !== undefined) {
        cell.style.color = column.sign(trade) < 0 ? THEME.down : THEME.up;
      }
      cell.appendChild(document.createTextNode(column.text(trade)));
      const time = column.locate === undefined ? null : column.locate(trade);
      if (time !== null) {
        addLocateButton(cell, market, time, priceChart);
      }
    });
  }

  function buildTradesView(container, market, priceChart) {
    const table = document.createElement("table");
    table.className = "trades";
    table.appendChild(buildTradeHead());

    const body = document.createElement("tbody");
    market.trades.forEach(function (trade) {
      buildTradeRow(body, trade, market, priceChart);
    });
    table.appendChild(body);
    container.replaceChildren(table);
  }

  function buildReportSection(section) {
    const element = document.createElement("div");
    element.className = "report-section";

    const title = document.createElement("div");
    title.className = "report-title";
    title.textContent = section.title;
    element.appendChild(title);

    const table = document.createElement("table");
    const body = document.createElement("tbody");
    section.rows.forEach(function (row) {
      const line = body.insertRow();
      line.insertCell().textContent = row.label;
      const value = line.insertCell();
      value.className = NUMERIC_CLASS;
      value.textContent = row.value;
    });
    table.appendChild(body);
    element.appendChild(table);
    return element;
  }

  function buildReportTable(table) {
    const element = document.createElement("div");
    element.className = "report-section report-table";

    const title = document.createElement("div");
    title.className = "report-title";
    title.textContent = table.title;
    element.appendChild(title);

    const grid = document.createElement("table");
    const head = grid.createTHead().insertRow();
    head.insertCell();
    table.columns.forEach(function (column) {
      const cell = head.insertCell();
      cell.className = NUMERIC_CLASS + " report-column";
      cell.textContent = column;
    });

    const body = document.createElement("tbody");
    table.groups.forEach(function (group) {
      if (group.title !== null) {
        const heading = body.insertRow();
        heading.className = "report-group";
        const cell = heading.insertCell();
        cell.colSpan = table.columns.length + 1;
        cell.textContent = group.title;
      }
      group.rows.forEach(function (row) {
        const line = body.insertRow();
        line.insertCell().textContent = row.label;
        row.values.forEach(function (value) {
          const cell = line.insertCell();
          cell.className = NUMERIC_CLASS;
          cell.textContent = value;
        });
      });
    });
    grid.appendChild(body);
    element.appendChild(grid);
    return element;
  }

  function buildConfigurationView(container, configuration) {
    const pre = document.createElement("pre");
    pre.className = "configuration";
    pre.textContent = configuration;
    container.appendChild(pre);
  }

  function buildReportView(container, report) {
    const element = document.createElement("div");
    element.className = "report";

    const stack = document.createElement("div");
    stack.className = "report-stack";
    report.sections.forEach(function (section) {
      stack.appendChild(buildReportSection(section));
    });
    element.appendChild(stack);
    element.appendChild(buildReportTable(report.trades));
    container.appendChild(element);
  }

  function createViewPanel() {
    const panel = document.createElement("div");
    panel.className = "panel view-panel";
    return panel;
  }

  function createViewBody(view) {
    const body = document.createElement("div");
    body.className = "view-body";
    body.dataset.view = view.label.toLowerCase();
    view.build(body);
    return body;
  }

  function selectTab(tabs, bodies, selected) {
    tabs.forEach(function (tab, index) {
      tab.setAttribute("aria-selected", String(index === selected));
    });
    bodies.forEach(function (body, index) {
      body.hidden = index !== selected;
    });
  }

  function tabViews(host, views) {
    const panel = createViewPanel();
    const bar = document.createElement("div");
    bar.className = "view-tabs";
    bar.setAttribute("role", "tablist");
    panel.appendChild(bar);

    const tabs = views.map(function (view) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "view-tab";
      tab.setAttribute("role", "tab");
      tab.textContent = view.label;
      bar.appendChild(tab);
      return tab;
    });
    const bodies = views.map(function (view) {
      const body = createViewBody(view);
      body.setAttribute("role", "tabpanel");
      panel.appendChild(body);
      return body;
    });

    tabs.forEach(function (tab, index) {
      tab.addEventListener("click", function () {
        selectTab(tabs, bodies, index);
      });
    });
    selectTab(tabs, bodies, 0);
    host.appendChild(panel);
  }

  function anyMarketHasTrades() {
    return payload.markets.some(function (market) {
      return market.trades.length > 0;
    });
  }

  function viewsOf(priceChart) {
    return [
      {
        label: "Report",
        present:
          payload.report.sections.length > 0 ||
          payload.report.trades.groups.length > 0,
        build: function (container) {
          buildReportView(container, payload.report);
        },
      },
      {
        label: "Trades",
        present: anyMarketHasTrades(),
        build: function (container) {
          buildTradesView(container, activeMarket(), priceChart);
        },
      },
      {
        label: "Configuration",
        present: payload.configuration !== "",
        build: function (container) {
          buildConfigurationView(container, payload.configuration);
        },
      },
    ];
  }

  function mountKeyStats() {
    const container = document.getElementById(KEY_STATS);
    payload.report.headline.forEach(function (figure) {
      container.appendChild(buildKeyStat(figure));
    });
  }

  function buildKeyStat(figure) {
    const stat = document.createElement("div");
    stat.className = "key-stat";

    const label = document.createElement("span");
    label.className = "key-stat-label";
    label.textContent = figure.label;
    stat.appendChild(label);

    const value = document.createElement("span");
    value.className = "key-stat-value";
    value.textContent = figure.value;
    stat.appendChild(value);
    return stat;
  }

  function mountViews(priceChart) {
    const host = document.getElementById("views");
    const views = viewsOf(priceChart).filter(function (view) {
      return view.present;
    });
    if (views.length === 0) {
      host.remove();
      return;
    }
    tabViews(host, views);
  }

  function refreshTradesView(priceChart) {
    const host = document.getElementById("views");
    if (host === null) {
      return;
    }
    const body = host.querySelector('[data-view="trades"]');
    if (body === null) {
      return;
    }
    buildTradesView(body, activeMarket(), priceChart);
  }

  function mountResizing() {
    const panes = [
      document.getElementById("price-panel"),
      document.getElementById(PERFORMANCE_KEY + "-panel"),
      document.getElementById("views"),
    ].filter(function (pane) {
      return pane !== null;
    });

    panes.slice(1).forEach(function (below, index) {
      addSplitter(panes[index], below);
    });
  }

  function addSplitter(above, below) {
    const bar = document.createElement("div");
    bar.className = "splitter";
    above.parentNode.insertBefore(bar, below);

    let grabbedAt = 0;
    let aboveHeight = 0;
    let belowHeight = 0;

    function drag(event) {
      const moved = heldWithinBounds(event.clientY - grabbedAt);
      setPaneHeight(above, aboveHeight + moved);
      setPaneHeight(below, belowHeight - moved);
    }

    function heldWithinBounds(moved) {
      const room = Math.min(moved, belowHeight - MIN_PANE_HEIGHT);
      return Math.max(room, MIN_PANE_HEIGHT - aboveHeight);
    }

    function release() {
      document.removeEventListener("mousemove", drag);
      document.removeEventListener("mouseup", release);
      document.body.classList.remove("resizing");
    }

    bar.addEventListener("mousedown", function (event) {
      grabbedAt = event.clientY;
      aboveHeight = above.getBoundingClientRect().height;
      belowHeight = below.getBoundingClientRect().height;
      document.body.classList.add("resizing");
      document.addEventListener("mousemove", drag);
      document.addEventListener("mouseup", release);
      event.preventDefault();
    });
  }

  function setPaneHeight(pane, height) {
    pane.style.flex = "0 0 " + height + "px";
    pane.style.minHeight = height + "px";
  }

  mountKeyStats();
  const price = buildPriceChart();
  const synchroniser = createSynchroniser(price.chart);
  mountPerformance(synchroniser);
  mountViews(price.chart);
  renderMarketMenu(function switchMarket(index) {
    activeMarketIndex = index;
    price.switchMarket(activeMarket());
    refreshTradesView(price.chart);
  });
  mountResizing();
})();
