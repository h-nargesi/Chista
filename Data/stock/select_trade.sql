use RahavardNovin3;
go

declare @ID int = 20038;
declare @Type char(1) = 'T';
declare @Offset bigint = 771;

--create or alter procedure GetTrade  @ID int, @Type char(1), @Offset bigint as

/*
RESULT_COUNT = 20;
SIGNAL_STEP_COUNT = 183;
SIGNAL_LAST_YEARS = 60;
YEARS_COUNT = 3;
*/

---- POINTER
with pointer as (
	select		lag(DateTimeEn, 20/*RESULT_COUNT*/) over (order by DateTimeEn desc) as StartDateEn
	from		Trade
	where		InstrumentID = @ID and RecordType = @Type
	order by	DateTimeEn desc
	offset		20/*RESULT_COUNT*/ + @Offset rows
	fetch		first 1 rows only

-- RESTRICTION
), ristriction as (
	select		StartDateEn, StartDateJl,
				(
		select		DateTimeEn
		from		Trade t
		where		InstrumentID = @ID and DateTimeEn < p_end.EndDateEn
		order by	DateTimeEn desc
		offset		60/*SIGNAL_LAST_YEARS*/ rows
		fetch		first 1 rows only
				) as EndDateEn
	from (
		select		StartDateEn, StartDateJl,
					dbo.jparse(StartDateJl / 10000 - 3,/*YEARS_COUNT*/
						StartDateJl % 10000 / 100,
						StartDateJl % 100, 0, 0, 0) as EndDateEn
		from (select StartDateEn, dbo.jalali(StartDateEn) as StartDateJl from pointer) p
	) p_end

-- STREAM
), stream as (
select		row_number() over (order by DateTimeEn desc) as Ranking,
			DateTimeEn, dbo.jalali(DateTimeEn) DateTimeJl,
			dbo.jalali(lead(DateTimeEn) over (order by DateTimeEn desc)) as DateTimeJlNext,
			StartDateJl,
			100 * isnull(ClosePriceChange / lead(ClosePrice) over (order by DateTimeEn desc), 0) as ChangePercent
from		Trade, ristriction
where		InstrumentID = @ID and DateTimeEn between EndDateEn and StartDateEn

-- DETAILS
), details as (
	select 	StartDateJl % 10000 as StartDateJl_Date,
			--DateTimeJl / 10000 as DateTimeJl_Year,
			--DateTimeJl % 10000 as DateTimeJl_Date,
			DateTimeJlNext / 10000 as DateTimeJlNext_Year,
			stream.*
	from   	stream

-- ANNUAL
), annual as (
  select 	Ranking, DateTimeEn,
			max(period_start) over (
				order by DateTimeEn desc rows between 59/*SIGNAL_LAST_YEARS-1*/ preceding and current row)
				as period_start,
			max(year_diff) over (
				order by DateTimeEn desc rows between 59/*SIGNAL_LAST_YEARS-1*/ preceding and current row)
				- 1 as year_sec_count,
			ChangePercent
    from (
      select 	case when DateTimeJlNext < Annual and Annual <= DateTimeJl then Ranking else null end as period_start,
				case when DateTimeJlNext < Annual and Annual <= DateTimeJl
					then (StartDateJl - Criterion_Year) / 10000 + 1 else null end as year_diff,
				annual_dtl.*
        from (
            select	Criterion_Year + StartDateJl_Date as Annual,
					dtl_criterion.*
            from (
				select	case when StartDateJl_Date > DateTimeJl % 10000
							then DateTimeJlNext_Year else DateTimeJl / 10000
						end * 10000 as Criterion_Year,
						details.*
				from	details
			) dtl_criterion
        ) annual_dtl
    ) annual_seq

-- LABEL
), label as (
    select	case
				when Ranking <= 20/*RESULT_COUNT*/
				then 0
				---------------------------------------------
				when year_sec_count > 0 then 10 + year_sec_count
				---------------------------------------------------------------
				when Ranking <= 20/*RESULT_COUNT*2*/ + 183/*SIGNAL_STEP_COUNT*/
				then 1
				----------------------------------------------------------------
				--when Ranking <= 40/*RESULT_COUNT*/ + 80/*SIGNAL_STEP_COUNT*2*/
				--then 2
				----------------------------------------------------------------
				--when Ranking <= 40/*RESULT_COUNT*/ + 120/*SIGNAL_STEP_COUNT*3*/
				--then 3
				----------------------------------------------------------------
				--when Ranking <= 40/*RESULT_COUNT*/ + 160/*SIGNAL_STEP_COUNT*4*/
				--then 4
				----------------------------------------------------------------
				else null
			end as Section,
			case
				when Ranking <= 20/*RESULT_COUNT*/
				then Ranking
				------------------------------------------------------------
				when year_sec_count > 0
				then floor((Ranking - period_start + year_sec_count) / year_sec_count)
				---------------------------------------------------------------
				when Ranking <= 20/*RESULT_COUNT*2*/ + 183/*SIGNAL_STEP_COUNT*/
				then (Ranking - 21) + 1
				----------------------------------------------------------------
				--when Ranking <= 40/*RESULT_COUNT*/ + 80/*SIGNAL_STEP_COUNT*2*/
				--then floor((Ranking - 61) / 2) + 1
				----------------------------------------------------------------
				--when Ranking <= 40/*RESULT_COUNT*/ + 120/*SIGNAL_STEP_COUNT*3*/
				--then floor((Ranking - 101) / 4) + 1
				----------------------------------------------------------------
				--when Ranking <= 40/*RESULT_COUNT*/ + 160/*SIGNAL_STEP_COUNT*4*/
				--then floor((Ranking - 141) / 8) + 1
				----------------------------------------------------------------
				else null
			end as Ranking,
			annual.DateTimeEn,
			annual.ChangePercent
      from	annual

-- SECTION
), section as (
    select p.Section
		 , p.Ranking
		 , avg(isnull(ChangePercent, 0)) as ChangePercent
		 
         , min(isnull(DateTimeEn, getdate())) as DateTimeEn
         , cast(p.Section as varchar) + '-' + cast(p.Ranking as varchar) as SectionString
         , count(*) as Quantity
		 
      from ristriction, Pattern p left join label
	    on p.Section = label.Section and p.Ranking = label.Ranking
  group by p.Section, p.Ranking
)

--select * from pointer
select * from label order by DateTimeEn desc 
--select ChangePercent from section order by Section, Ranking
