-- db/migrations/001_create_leads.sql
-- Phase 6: leads table. Columns are the flattened fields of models/lead.py
-- (LeadRecord = LeadInput + LeadScore + delivery flags).
-- Run once in Supabase Dashboard > SQL Editor. Safe to re-run: it never alters an
-- existing table. Schema changes go in a new file (002_...).
--
-- Conventions the schema-contract test relies on:
--   * one column per line, indented two spaces, inside the CREATE TABLE block
--   * table-level constraints start with the keyword "constraint"
--
-- NOTE: category is NOT derived from score here. The 40/70 thresholds live only in
-- models/lead.py (WARM_SCORE_MIN / HOT_SCORE_MIN); duplicating them in SQL would let
-- the two drift apart.

create table if not exists public.leads (
  -- LeadRecord metadata
  id                 uuid        primary key default gen_random_uuid(),
  created_at         timestamptz not null default now(),

  -- LeadInput (never updated after insert)
  name               text        not null,
  email              text        not null,
  phone              text        not null,
  message            text        not null,
  property_type      text,
  preferred_location text,
  budget             text,
  timeline           text,

  -- LeadScore (never updated after insert)
  score              smallint    not null,
  category           text        not null,
  reasoning          text        not null,
  suggested_action   text        not null,
  priority_level     text        not null,

  -- Idempotency flags: the service only ever sets these false -> true
  email_sent         boolean     not null default false,
  sms_sent           boolean     not null default false,
  hubspot_created    boolean     not null default false,

  constraint leads_name_not_blank    check (char_length(btrim(name)) > 0),
  constraint leads_phone_min_length  check (char_length(btrim(phone)) >= 7),
  constraint leads_message_not_blank check (char_length(btrim(message)) > 0),
  constraint leads_score_range       check (score between 0 and 100),
  constraint leads_category_allowed  check (category in ('hot', 'warm', 'cold')),
  constraint leads_priority_allowed  check (priority_level in ('urgent', 'normal', 'low'))
);

create index if not exists leads_created_at_idx  on public.leads (created_at desc);
create index if not exists leads_email_lower_idx on public.leads (lower(email));

comment on table public.leads is
  'AI-scored real estate leads (personal data). Access: service_role only; RLS on, no policies.';

-- ACCESS CONTROL ------------------------------------------------------------
-- Layer 1, grants: since 2026-10-30 new public tables are not reachable through the
--   Data API until a role is granted access explicitly.
-- Layer 2, RLS: enabled with NO policies, so anon/authenticated can never read a row,
--   even if a grant is added by mistake. service_role bypasses RLS but still needs grants.
alter table public.leads enable row level security;

revoke all on table public.leads from anon, authenticated, service_role;
grant select, insert, update on table public.leads to service_role;
-- DELETE is intentionally not granted: the backend never deletes leads.
-- Remove rows from the SQL Editor, which runs as a privileged role.

notify pgrst, 'reload schema';