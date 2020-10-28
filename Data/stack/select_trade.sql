use RahavardNovin3;
go

declare @ID int = 2892;
declare @Type char(1) = 'X';
declare @Offset bigint = 10;

--create or alter procedure GetTrade  @ID int, @Type char(1), @Offset bigint as

/*
RESULT_COUNT = 20;
SIGNAL_STEP_COUNT = 40;
SIGNAL_LAST_YEARS = SIGNAL_STEP_COUNT + RESULT_COUNT;
YEARS_COUNT = 3;
*/

---- POINTER
with pointer as (
	select		lag(DateTimeEn, 20/*RESULT_COUNT*/) over (order by DateTimeEn desc) as StartDateEn
	from		Trade
	where		InstrumentID = @ID and RecordType = @Type
	order by	DateTimeEn desc offset 20/*RESULT_COUNT*/ + @Offset rows fetch first 1 rows only

-- RESTRICTION
), ristriction as (
	select		StartDateEn,
				dateadd(month, 1,
					dbo.jparse(StartDateJl / 10000 - 4, -- YEARS_COUNT + 1
					StartDateJl % 10000 / 100,
					StartDateJl % 100, 0, 0, 0)) as EndDateEn
	from (select StartDateEn, dbo.jalali(StartDateEn) as StartDateJl from pointer) p

-- STREAM
), stream as (
	select		strm.*,
				dbo.jalali(DateTimeEnNext) as DateTimeJlNext,
				floor(DateTimeJl / 10000) as DateTimeJYear
	from (
		select		row_number() over (order by DateTimeEn desc) as Ranking,
					StartDateEn, dbo.jalali(StartDateEn) as StartDateJl,
					DateTimeEn, dbo.jalali(DateTimeEn) as DateTimeJl,
					lead(DateTimeEn) over (order by DateTimeEn desc) as DateTimeEnNext,
					100 * isnull(ClosePriceChange / lead(ClosePrice) over (order by DateTimeEn desc), 0) as ChangePercent
		from		Trade, ristriction
		where		InstrumentID = @ID and RecordType is not null and DateTimeEn between EndDateEn and StartDateEn
	) strm

-- ANNUAL
), annual as (
  select Ranking, DateTimeEn,
         max(period_start) over (order by DateTimeEn desc rows between 60/*SIGNAL_LAST_YEARS*/ preceding and current row)
			as period_start,
         max(year_diff) over (order by DateTimeEn desc rows between 60/*SIGNAL_LAST_YEARS*/ preceding and current row)
			as year_diff,
         ChangePercent
    from (
      select case when Annual between DateTimeJlNext and DateTimeJl then Ranking else null end as period_start,
             case when Annual between DateTimeJlNext and DateTimeJl then StartDateJl / 10000 - DateTimeJYear + 1 else null end as year_diff,
             annual.*
        from (
            select DateTimeJYear * 10000 + (StartDateJl % 10000) as Annual,
                   stream.*
              from stream
        ) annual
    ) annual_seq

-- LABEL
), label as (
    select case
             when Ranking <= 20/*RESULT_COUNT*/								then '0-' + cast(Ranking as varchar)
             when Ranking <= 20/*RESULT_COUNT*/ + 40/*SIGNAL_STEP_COUNT*/	then '1-' + cast(Ranking as varchar)
             when Ranking <= 20/*RESULT_COUNT*/ + 80/*SIGNAL_STEP_COUNT*/	then '2-' + cast(floor(Ranking / 2) as varchar)
             when Ranking <= 20/*RESULT_COUNT*/ + 120/*SIGNAL_STEP_COUNT*/	then '3-' + cast(floor(Ranking / 4) as varchar)
             when Ranking <= 20/*RESULT_COUNT*/ + 160/*SIGNAL_STEP_COUNT*/	then '4-' + cast(floor(Ranking / 8) as varchar)
             ------------------------------------------------------------------------------------------
			 --when year_diff > 3 then null
             when Ranking between period_start and period_start + 60/*SIGNAL_LAST_YEARS*/ then 
                  '7-' + cast(year_diff as varchar) + '-' + cast(floor(Ranking / year_diff) as varchar)
             ------------------------------------------------------------------------------------------
             else null
           end as Section,
           annual.*
      from annual

-- SECTION
), section as (
    select --Section,
           --min(Ranking) as Ranking,
           min(DateTimeEn) as DateTimeEn,
           avg(ChangePercent) as ChangePercent
      from label
     where Section is not null
  group by Section
)

select ChangePercent from section order by DateTimeEn desc
