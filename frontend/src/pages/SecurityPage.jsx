import { useState, useEffect, useCallback } from 'react'
import {
  Shield, ShieldAlert, ShieldCheck, Users, Key, Lock,
  AlertTriangle, CheckCircle2, XCircle, RefreshCw, Activity,
  Eye, EyeOff, ChevronDown, ChevronRight, Cpu, Database,
  Clock, User, Hash, Zap,
} from 'lucide-react'
import { securityAPI } from '../api/endpoints'
import { useAuthStore } from '../store/authStore'
import toast from 'react-hot-toast'

/* ─── Severity badge ─── */
function SeverityBadge({ severity }) {
  const map = {
    CRITICAL: 'bg-rose-500/15 text-rose-400 border-rose-500/25',
    WARNING:  'bg-amber-500/15 text-amber-400 border-amber-500/25',
    INFO:     'bg-cyan-500/15  text-cyan-400  border-cyan-500/25',
  }
  const cls = map[severity] || map.INFO
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border uppercase tracking-wider ${cls}`}>
      {severity}
    </span>
  )
}

/* ─── Event type label ─── */
function EventTypeBadge({ type }) {
  const map = {
    login_success:   { label: 'Login OK',       cls: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
    login_failed:    { label: 'Login Failed',    cls: 'bg-rose-500/10    text-rose-400    border-rose-500/20' },
    register:        { label: 'Register',        cls: 'bg-violet-500/10  text-violet-400  border-violet-500/20' },
    rbac_denied:     { label: 'RBAC Denied',     cls: 'bg-amber-500/10   text-amber-400   border-amber-500/20' },
    token_expired:   { label: 'Token Expired',   cls: 'bg-amber-500/10   text-amber-400   border-amber-500/20' },
    token_invalid:   { label: 'Invalid Token',   cls: 'bg-rose-500/10    text-rose-400    border-rose-500/20' },
    injection_blocked: { label: 'Injection Blocked', cls: 'bg-rose-500/10 text-rose-400   border-rose-500/20' },
    role_changed:    { label: 'Role Changed',    cls: 'bg-indigo-500/10  text-indigo-400  border-indigo-500/20' },
  }
  const m = map[type] || { label: type, cls: 'bg-surface-container text-on-surface-variant border-outline-variant/20' }
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border ${m.cls}`}>
      {m.label}
    </span>
  )
}

/* ─── Stat card ─── */
function StatCard({ icon: Icon, label, value, sub, color = 'text-primary' }) {
  return (
    <div className="bg-surface-container rounded-xl border border-outline-variant/10 p-4 flex items-center gap-4">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center bg-surface-container-highest ${color}`}>
        <Icon size={20} strokeWidth={1.8} />
      </div>
      <div>
        <p className="text-2xl font-bold text-on-surface leading-none">{value}</p>
        <p className="text-xs text-on-surface-variant mt-0.5">{label}</p>
        {sub && <p className="text-[10px] text-on-surface-variant/50 mt-0.5">{sub}</p>}
      </div>
    </div>
  )
}

/* ─── Module status dot ─── */
function ModuleDot({ status }) {
  if (status === 'active') return <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
  return <span className="w-2 h-2 rounded-full bg-rose-400 inline-block" />
}

/* ══════════════ TABS ══════════════ */

/* Tab 1: Security Events */
function EventsTab() {
  const [events, setEvents]     = useState([])
  const [stats, setStats]       = useState(null)
  const [loading, setLoading]   = useState(true)
  const [filter, setFilter]     = useState('all')  // all | WARNING | CRITICAL

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = filter !== 'all' ? { severity: filter } : {}
      const [evRes, stRes] = await Promise.all([
        securityAPI.events(params),
        securityAPI.stats(),
      ])
      setEvents(evRes.data)
      setStats(stRes.data)
    } catch {
      toast.error('Failed to load security events')
    } finally {
      setLoading(false)
    }
  }, [filter])

  useEffect(() => { load() }, [load])

  return (
    <div className="space-y-5">
      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatCard icon={Activity}   label="Total Events"   value={stats.total_events}   color="text-cyan-400" />
          <StatCard icon={AlertTriangle} label="Warnings"    value={stats.warning_count}  color="text-amber-400" sub="Last recorded" />
          <StatCard icon={ShieldAlert} label="Critical"      value={stats.critical_count} color="text-rose-400" />
          <StatCard icon={Clock}       label="Last 24 h"     value={stats.last_24h}        color="text-violet-400" />
        </div>
      )}

      {/* By-type breakdown */}
      {stats?.by_type && Object.keys(stats.by_type).length > 0 && (
        <div className="bg-surface-container rounded-xl border border-outline-variant/10 p-4">
          <p className="text-xs font-bold text-on-surface-variant uppercase tracking-wider mb-3">Events by type</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(stats.by_type).map(([type, count]) => (
              <span key={type} className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface-container-highest text-xs font-medium text-on-surface">
                <EventTypeBadge type={type} />
                <span className="text-on-surface-variant ml-1">×{count}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Filter bar */}
      <div className="flex items-center gap-2">
        <p className="text-xs font-semibold text-on-surface-variant mr-2">Filter:</p>
        {['all', 'INFO', 'WARNING', 'CRITICAL'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              filter === f
                ? 'bg-primary/15 text-primary border border-primary/25'
                : 'bg-surface-container text-on-surface-variant border border-outline-variant/10 hover:bg-surface-container-highest'
            }`}
          >
            {f === 'all' ? 'All' : f}
          </button>
        ))}
        <button
          onClick={load}
          disabled={loading}
          className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-surface-container text-on-surface-variant border border-outline-variant/10 hover:bg-surface-container-highest transition-colors"
        >
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Events table */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/10 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-16 text-on-surface-variant">
            <RefreshCw size={20} className="animate-spin mr-2" /> Loading…
          </div>
        ) : events.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-on-surface-variant gap-2">
            <ShieldCheck size={32} className="text-emerald-400 mb-1" />
            <p className="font-semibold">No events recorded yet</p>
            <p className="text-xs">Events appear here after logins, registrations, or security violations.</p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-outline-variant/10">
                <th className="text-left px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant/60">Time</th>
                <th className="text-left px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant/60">Type</th>
                <th className="text-left px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant/60">Severity</th>
                <th className="text-left px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant/60">User</th>
                <th className="text-left px-4 py-3 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant/60">Detail</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={e.id} className={`border-b border-outline-variant/5 hover:bg-surface-container-highest/50 transition-colors ${i % 2 === 0 ? '' : 'bg-surface-container-highest/20'}`}>
                  <td className="px-4 py-3 text-xs text-on-surface-variant font-mono whitespace-nowrap">
                    {new Date(e.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3"><EventTypeBadge type={e.event_type} /></td>
                  <td className="px-4 py-3"><SeverityBadge severity={e.severity} /></td>
                  <td className="px-4 py-3 text-xs font-semibold text-on-surface">{e.username || <span className="text-on-surface-variant/40 italic">anonymous</span>}</td>
                  <td className="px-4 py-3 text-xs text-on-surface-variant font-mono max-w-xs truncate">
                    {Object.entries(e.detail || {}).map(([k, v]) => `${k}: ${v}`).join(' · ') || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

/* Tab 2: Users & Hashed Passwords */
function UsersTab() {
  const [users, setUsers]       = useState([])
  const [loading, setLoading]   = useState(true)
  const [showHash, setShowHash] = useState({})

  useEffect(() => {
    securityAPI.usersSnapshot()
      .then(r => setUsers(r.data))
      .catch(() => toast.error('Failed to load users snapshot'))
      .finally(() => setLoading(false))
  }, [])

  const toggleHash = (id) => setShowHash(prev => ({ ...prev, [id]: !prev[id] }))

  const ROLE_COLOR = {
    admin:   'text-violet-400 bg-violet-500/10 border-violet-500/20',
    analyst: 'text-cyan-400   bg-cyan-500/10   border-cyan-500/20',
    viewer:  'text-amber-400  bg-amber-500/10  border-amber-500/20',
  }

  return (
    <div className="space-y-5">
      <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl px-4 py-3 flex items-start gap-3">
        <Lock size={16} className="text-amber-400 mt-0.5 shrink-0" />
        <div className="text-xs text-amber-300">
          <span className="font-bold">Security proof:</span> Passwords are stored as bcrypt hashes (cost=12). Raw passwords are never stored. Even admins cannot reverse them.
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-16 text-on-surface-variant">
          <RefreshCw size={20} className="animate-spin mr-2" /> Loading…
        </div>
      ) : (
        <div className="space-y-3">
          {users.map(u => (
            <div key={u.id} className="bg-surface-container rounded-xl border border-outline-variant/10 p-4">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-8 h-8 rounded-lg bg-primary/15 border border-primary/20 flex items-center justify-center">
                  <span className="text-xs font-bold text-primary">{u.username.slice(0,2).toUpperCase()}</span>
                </div>
                <div className="flex-1">
                  <p className="text-sm font-bold text-on-surface">{u.username}</p>
                  <p className="text-xs text-on-surface-variant">{u.email || 'No email'}</p>
                </div>
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10px] font-bold uppercase ${ROLE_COLOR[u.role] || ROLE_COLOR.viewer}`}>
                  {u.role}
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2 text-xs">
                <div className="flex items-start gap-2 bg-surface-container-highest rounded-lg px-3 py-2">
                  <Hash size={12} className="text-on-surface-variant mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <span className="text-on-surface-variant/60 font-semibold uppercase tracking-wider text-[10px]">Algorithm: </span>
                    <span className="text-emerald-400 font-bold">{u.pw_algorithm}</span>
                  </div>
                </div>

                <div className="flex items-start gap-2 bg-surface-container-highest rounded-lg px-3 py-2">
                  <Key size={12} className="text-on-surface-variant mt-0.5 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-on-surface-variant/60 font-semibold uppercase tracking-wider text-[10px]">Stored hash</span>
                      <button
                        onClick={() => toggleHash(u.id)}
                        className="flex items-center gap-1 text-[10px] text-primary hover:text-primary/80 transition-colors"
                      >
                        {showHash[u.id] ? <EyeOff size={10} /> : <Eye size={10} />}
                        {showHash[u.id] ? 'Hide' : 'Reveal'}
                      </button>
                    </div>
                    {showHash[u.id] ? (
                      <code className="text-cyan-400 break-all font-mono text-[11px] leading-relaxed">{u.hashed_pw}</code>
                    ) : (
                      <span className="text-on-surface-variant font-mono">{'•'.repeat(60)}</span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 bg-surface-container-highest rounded-lg px-3 py-2">
                  <Clock size={12} className="text-on-surface-variant shrink-0" />
                  <span className="text-on-surface-variant/60 font-semibold uppercase tracking-wider text-[10px]">Registered: </span>
                  <span className="text-on-surface text-xs">{new Date(u.created_at).toLocaleString()}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* Tab 3: Module Status */
function ModulesTab() {
  const [modules, setModules]   = useState(null)
  const [loading, setLoading]   = useState(true)

  const MODULE_META = {
    jwt_handler:       { label: 'JWT Handler',         desc: 'HS256 token creation & verification (PyJWT)',           icon: Key },
    password_hashing:  { label: 'Password Hashing',    desc: 'bcrypt with cost factor 12 — one-way, salted',         icon: Lock },
    rbac:              { label: 'RBAC',                 desc: 'Role-Based Access Control (admin / analyst / viewer)',  icon: Shield },
    prompt_injection:  { label: 'Prompt Injection Guard', desc: 'Pattern-based + ML scoring — blocks/flags injections', icon: ShieldAlert },
    sanitization:      { label: 'Input Sanitization',   desc: 'Strips dangerous characters and sequences',            icon: ShieldCheck },
    aes_encryption:    { label: 'AES Encryption',       desc: 'AES-256 for at-rest data encryption',                 icon: Database },
    audit_logger:      { label: 'Audit Logger',         desc: 'Structured JSON event logs (file + DB)',               icon: Activity },
    file_upload_guard: { label: 'File Upload Guard',    desc: 'MIME type check, size limits, malware patterns',       icon: Zap },
  }

  useEffect(() => {
    // Use /security/status from main.py (no auth needed)
    import('../api/client').then(({ default: api }) => {
      api.get('/security/status')
        .then(r => setModules(r.data.security_modules))
        .catch(() => toast.error('Failed to load module status'))
        .finally(() => setLoading(false))
    })
  }, [])

  const active  = modules ? Object.values(modules).filter(v => v === 'active').length : 0
  const total   = modules ? Object.keys(modules).length : 0

  return (
    <div className="space-y-5">
      {!loading && modules && (
        <div className="bg-surface-container rounded-xl border border-outline-variant/10 p-4 flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <ShieldCheck size={26} className="text-emerald-400" />
          </div>
          <div>
            <p className="text-2xl font-bold text-emerald-400">{active}/{total}</p>
            <p className="text-sm text-on-surface-variant">Security modules active</p>
          </div>
          <div className="ml-auto">
            <div className="w-32 h-2 bg-surface-container-highest rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-400 rounded-full transition-all"
                style={{ width: `${(active / total) * 100}%` }}
              />
            </div>
            <p className="text-xs text-on-surface-variant/60 mt-1 text-right">{Math.round((active/total)*100)}% operational</p>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-16 text-on-surface-variant">
          <RefreshCw size={20} className="animate-spin mr-2" /> Loading…
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {modules && Object.entries(modules).map(([key, status]) => {
            const meta   = MODULE_META[key] || { label: key, desc: '', icon: Cpu }
            const Icon   = meta.icon
            const active = status === 'active'
            return (
              <div
                key={key}
                className={`bg-surface-container rounded-xl border p-4 flex items-start gap-3 ${
                  active ? 'border-emerald-500/15' : 'border-rose-500/20'
                }`}
              >
                <div className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
                  active ? 'bg-emerald-500/10' : 'bg-rose-500/10'
                }`}>
                  <Icon size={18} className={active ? 'text-emerald-400' : 'text-rose-400'} strokeWidth={1.8} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-bold text-on-surface">{meta.label}</p>
                    <ModuleDot status={status} />
                    <span className={`text-[10px] font-bold uppercase ${active ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {active ? 'active' : 'error'}
                    </span>
                  </div>
                  <p className="text-xs text-on-surface-variant mt-0.5">{meta.desc}</p>
                  {!active && (
                    <p className="text-[10px] text-rose-400 mt-1 font-mono break-all">{status}</p>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* JWT architecture note */}
      <div className="bg-surface-container rounded-xl border border-outline-variant/10 p-4">
        <p className="text-xs font-bold text-on-surface-variant uppercase tracking-wider mb-3">JWT Token Structure</p>
        <div className="font-mono text-xs bg-surface-container-highest rounded-lg p-3 space-y-1">
          <p><span className="text-rose-400">HEADER</span><span className="text-on-surface-variant">.</span><span className="text-amber-400">PAYLOAD</span><span className="text-on-surface-variant">.</span><span className="text-emerald-400">SIGNATURE</span></p>
          <p className="text-on-surface-variant">╠ Algorithm: <span className="text-cyan-400">HS256 (HMAC-SHA256)</span></p>
          <p className="text-on-surface-variant">╠ Claims: <span className="text-cyan-400">user_id, username, role, iat, exp</span></p>
          <p className="text-on-surface-variant">╠ Expiry: <span className="text-cyan-400">30 minutes</span></p>
          <p className="text-on-surface-variant">╚ Secret: <span className="text-cyan-400">min 32-char key from JWT_SECRET_KEY env</span></p>
        </div>
      </div>
    </div>
  )
}

/* ══════════════ MAIN PAGE ══════════════ */

const TABS = [
  { id: 'events',  label: 'Security Events', icon: Activity },
  { id: 'users',   label: 'Users & Auth',    icon: Users },
  { id: 'modules', label: 'Module Status',   icon: Cpu },
]

export default function SecurityPage() {
  const { user } = useAuthStore()
  const [tab, setTab] = useState('events')

  if (user?.role !== 'admin') {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-on-surface-variant">
        <ShieldAlert size={48} className="text-rose-400" />
        <p className="text-lg font-bold text-on-surface">Admin access required</p>
        <p className="text-sm">Only admins can view the security dashboard.</p>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto py-6 px-4 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-2xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
          <Shield size={24} className="text-rose-400" strokeWidth={1.8} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-on-surface">Security Monitor</h1>
          <p className="text-sm text-on-surface-variant">Live view of authentication, access control, and threat detection</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-outline-variant/10">
        {TABS.map(t => {
          const Icon = t.icon
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-semibold border-b-2 transition-colors ${
                tab === t.id
                  ? 'border-primary text-primary'
                  : 'border-transparent text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <Icon size={15} strokeWidth={tab === t.id ? 2.2 : 1.8} />
              {t.label}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div>
        {tab === 'events'  && <EventsTab />}
        {tab === 'users'   && <UsersTab />}
        {tab === 'modules' && <ModulesTab />}
      </div>
    </div>
  )
}
