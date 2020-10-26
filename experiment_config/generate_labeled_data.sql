/*
-- create temp table

create table public.bill_dates
(
    bill_id int,
    introduced_date date,
    final_date date
)


-- add bills which were passed

insert into public.bill_dates

select bp.bill_id as bill_id, b.introduced_date as introduced_date, bp.progress_date as final_date
from (
	select bill_id, progress_date 
	from ml_policy_class.bill_progress where bill_status = 4
) bp
join ml_policy_class.bills b on b.bill_id = bp.bill_id 
join ml_policy_class.sessions s on s.session_id = b.session_id 
where s.state_id = 32


-- add bills which were not passed

insert into public.bill_dates
(	select bill_id, introduced_date from public.bill_dates
    except
    select bill_id, introduced_date from ml_policy_class.bills)  
union all
(   select bill_id, introduced_date from (
		select b.bill_id as bill_id, b.introduced_date as introduced_date
		from ml_policy_class.bills b
		join ml_policy_class.sessions s on s.session_id = b.session_id 
		where s.state_id = 32
	) bb
    except
    select bill_id, introduced_date from public.bill_dates)


-- add session id

alter table public.bill_dates
add column session_id int

update public.bill_dates 
set session_id = b.session_id
from ml_policy_class.bills b
where public.bill_dates.bill_id = b.bill_id 


-- update final date as session end date for not passed bills

alter table public.bill_dates
add column session_end_date date

update public.bill_dates
set session_end_date = '2012-06-22'
where  public.bill_dates.session_id = 93

update public.bill_dates
set session_end_date = '2014-06-23'
where  public.bill_dates.session_id = 1013

update public.bill_dates
set session_end_date = '2016-06-18'
where  public.bill_dates.session_id = 1143

update public.bill_dates
set session_end_date = '2018-06-20'
where  public.bill_dates.session_id = 1420

update public.bill_dates
set session_end_date = '2020-12-31'
where  public.bill_dates.session_id = 1644


update public.bill_dates
set final_date = session_end_date
where public.bill_dates.final_date is null


-- create new permanent table for processed data

create table sketch.bill_processed
(
    bill_id int,
    session_id int,
    introduced_date date,
    final_date date,
    present_date date
)


-- create rows with current date between introduced and final dates

insert into sketch.bill_processed
select aa.bill_id  as bill_id, aa.session_id as session_id, aa.introduced_date as introduced_date, aa.final_date as final_date, cast(aa.present_date as date) as present_date
from (
	select bd.bill_id  as bill_id, bd.session_id as session_id, bd.introduced_date as introduced_date, bd.final_date as final_date, cast(present_date.present_date as date)
	from public.bill_dates bd 
	join generate_series('2009-01-07', '2020-12-31', interval '1 day') present_date on present_date.present_date >= bd.introduced_date
	and present_date.present_date <= bd.final_date and bd.final_date is not null
) aa
--group by bd.bill_id, bd.introduced_date, bd.final_date, bd.session_id, bd.session_end_date, present_date


-- add column for session end date

alter table sketch.bill_processed
add column session_end_date date

update sketch.bill_processed
set session_end_date = bd.session_end_date
from public.bill_dates bd
where sketch.bill_processed.bill_id = bd.bill_id


-- add column for labels

alter table sketch.bill_processed
add column label int

update sketch.bill_processed
set label = 0

update sketch.bill_processed
set label = 1
where (date_part('year', sketch.bill_processed.final_date) - date_part('year', sketch.bill_processed.present_date)) * 12 +
             (date_part('month', sketch.bill_processed.final_date) - date_part('month', sketch.bill_processed.present_date)) <= 6
and sketch.bill_processed.final_date != sketch.bill_processed.session_end_date


alter table sketch.bill_processed 
drop column session_end_date

**/











