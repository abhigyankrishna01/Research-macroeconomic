'use client';
import { useEffect, useState } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';
import { ShieldAlert, Cpu, Database, Ban } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

type TransactionAlert = {
  transaction_id: string;
  user_id: string;
  amount: number;
  merchant_category: string;
  location: string;
  timestamp: string;
  risk_score: number;
  rules_triggered?: string[];
  status?: string;
  malicious_tally?: number;
  is_fraud?: boolean | null;
};

type ChartPoint = {
  name: string;
  score: number;
};

export default function Dashboard() {
  const [transactions, setTransactions] = useState<TransactionAlert[]>([]);
  const [metrics, setMetrics] = useState({ totalAnalysed: 0, criticalAlerts: 0, autoBlocked: 0 });
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [expandedTxId, setExpandedTxId] = useState<string | null>(null);
  const [labelingTx, setLabelingTx] = useState<Record<string, boolean>>({});
  const [reportLoading, setReportLoading] = useState(false);
  const [criticalFilterActive, setCriticalFilterActive] = useState(false);

  const API_URL = 'http://127.0.0.1:8001';
  const WS_URL = 'ws://127.0.0.1:8001/ws/alerts';
  const { lastJsonMessage, readyState } = useWebSocket(WS_URL, {
    shouldReconnect: () => true,
  });

  // Load historical transactions on mount so the feed is never empty after a refresh
  useEffect(() => {
    fetch(`${API_URL}/api/transactions/recent?limit=50`)
      .then(r => r.json())
      .then((data: TransactionAlert[]) => {
        setTransactions(data);
        const criticals = data.filter(t => t.risk_score >= 80).length;
        const blocked = data.filter(t => t.status === 'BLOCKED').length;
        setMetrics({ totalAnalysed: data.length, criticalAlerts: criticals, autoBlocked: blocked });
        const points = data.slice(-15).map(t => ({
          name: new Date(t.timestamp).toLocaleTimeString(),
          score: t.risk_score,
        }));
        setChartData(points);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (lastJsonMessage) {
      const msg = lastJsonMessage as TransactionAlert;
      setTransactions(prev => [msg, ...prev].slice(0, 50));
      setMetrics(prev => {
        let alerts = prev.criticalAlerts;
        let blocked = prev.autoBlocked;
        if (msg.risk_score >= 80) alerts++;
        if (msg.status === 'BLOCKED') blocked++;

        return {
          totalAnalysed: prev.totalAnalysed + 1,
          criticalAlerts: alerts,
          autoBlocked: blocked
        };
      });
      setChartData(prev => {
        const newData = [...prev, { name: new Date().toLocaleTimeString(), score: msg.risk_score }];
        return newData.slice(-15);
      });
    }
  }, [lastJsonMessage]);

  const handleLabel = async (transactionId: string, isFraud: boolean) => {
    setLabelingTx(prev => ({ ...prev, [transactionId]: true }));
    try {
      const response = await fetch(`${API_URL}/api/transactions/${transactionId}/label`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_fraud: isFraud }),
      });

      if (!response.ok) {
        throw new Error('Label update failed');
      }

      setTransactions(prev =>
        prev.map(tx => (tx.transaction_id === transactionId ? { ...tx, is_fraud: isFraud } : tx))
      );
    } catch (error) {
      console.error(error);
      alert('Unable to save analyst label right now.');
    } finally {
      setLabelingTx(prev => ({ ...prev, [transactionId]: false }));
    }
  };

  const generateReport = async () => {
    setReportLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/reports/summary?download=true`);
      if (!response.ok) {
        throw new Error('Failed to generate report');
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fraud-report-${new Date().toISOString()}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);
      alert('Report generation failed.');
    } finally {
      setReportLoading(false);
    }
  };

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting...',
    [ReadyState.OPEN]: 'Live Stream Active',
    [ReadyState.CLOSING]: 'Closing...',
    [ReadyState.CLOSED]: 'Connection Lost. Reconnecting...',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];
  const visibleTransactions = criticalFilterActive
    ? transactions.filter((tx) => tx.risk_score >= 80)
    : transactions;

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1 className="title">Fraud Detector platform</h1>
        <div className="header-actions">
          <button className="report-button" onClick={generateReport} disabled={reportLoading}>
            {reportLoading ? 'Generating...' : 'Generate Final Report'}
          </button>
          <div className="live-badge">
            <div className="pulse"></div>
            {connectionStatus}
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="metrics-grid">
          <div className="glass-panel metric-card">
            <span className="metric-label">Analyzed Velocity</span>
            <div className="metric-value">{metrics.totalAnalysed}</div>
            <Cpu style={{ color: "var(--accent-blue)" }} size={28} />
          </div>
          <button
            type="button"
            className={`glass-panel metric-card metric-card-button ${criticalFilterActive ? 'metric-card-active' : ''}`}
            onClick={() => setCriticalFilterActive((prev) => !prev)}
          >
            <span className="metric-label">Critical Alerts</span>
            <div className="metric-value" style={{ color: "var(--accent-red)" }}>{metrics.criticalAlerts}</div>
            <ShieldAlert style={{ color: "var(--accent-red)" }} size={28} />
          </button>
          <div className="glass-panel metric-card">
            <span className="metric-label">Auto-Blocked</span>
            <div className="metric-value">{metrics.autoBlocked}</div>
            <Ban size={28} />
          </div>
        </section>

        <section className="glass-panel chart-container">
          <h2 style={{ marginBottom: "1rem" }}>Live Risk Topology</h2>
          <ResponsiveContainer width="100%" height="80%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} />
              <YAxis stroke="var(--text-muted)" fontSize={12} />
              <Tooltip contentStyle={{ background: "var(--panel-bg)", border: "none", borderRadius: "8px" }} />
              <Area type="monotone" dataKey="score" stroke="var(--accent-cyan)" fill="url(#colorScore)" />
            </AreaChart>
          </ResponsiveContainer>
        </section>
      </main>

      <aside className="glass-panel">
        <div className="feed-header">
          <h2>
            {criticalFilterActive ? 'Critical Alert Transactions' : 'Live Feed'}
          </h2>
          {criticalFilterActive && (
            <button
              type="button"
              className="clear-filter-button"
              onClick={() => setCriticalFilterActive(false)}
            >
              Show All
            </button>
          )}
        </div>
        <div className="feed-list">
          {visibleTransactions.map((tx, i) => {
            const isCritical = tx.risk_score >= 80;
            return (
              <div
                key={i}
                className={`transaction-card ${isCritical ? 'critical' : ''}`}
                onClick={() =>
                  setExpandedTxId(prev => (prev === tx.transaction_id ? null : tx.transaction_id))
                }
              >
                <div className="tx-header">
                  <span>{tx.merchant_category.toUpperCase()}</span>
                  <span className={`tx-score ${isCritical ? 'score-high' : tx.risk_score > 40 ? 'score-medium' : 'score-low'}`}>
                    Risk: {tx.risk_score}
                  </span>
                </div>
                <div className="tx-amount">${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2 })}</div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>TXID: {tx.transaction_id.slice(0, 8)}...</div>
                {expandedTxId === tx.transaction_id && (
                  <div className="tx-expanded">
                    <div><strong>Location:</strong> {tx.location}</div>
                    <div><strong>Exact Time:</strong> {new Date(tx.timestamp).toLocaleString()}</div>
                    <div><strong>Status:</strong> {tx.status || 'PROCESSED'}</div>
                    <div><strong>Trigger Traces:</strong> {(tx.rules_triggered || []).join(', ') || 'None'}</div>
                    <div><strong>Malicious Tally:</strong> {tx.malicious_tally ?? 0}</div>
                    <div><strong>Analyst Label:</strong> {tx.is_fraud === undefined || tx.is_fraud === null ? 'Pending review' : tx.is_fraud ? 'True Fraud' : 'False Positive'}</div>
                    <div className="tx-actions" onClick={(e) => e.stopPropagation()}>
                      <button
                        className="label-button fraud"
                        disabled={Boolean(labelingTx[tx.transaction_id])}
                        onClick={() => handleLabel(tx.transaction_id, true)}
                      >
                        Mark True Fraud
                      </button>
                      <button
                        className="label-button legit"
                        disabled={Boolean(labelingTx[tx.transaction_id])}
                        onClick={() => handleLabel(tx.transaction_id, false)}
                      >
                        Dismiss / False Positive
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
          {visibleTransactions.length === 0 && (
            <div style={{ textAlign: "center", color: "var(--text-muted)", marginTop: "2rem" }}>
              <Database size={32} style={{ margin: "0 auto", opacity: 0.5, marginBottom: "0.5rem" }} />
              {criticalFilterActive ? 'No critical alerts in current feed.' : 'Awaiting Transactions...'}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
