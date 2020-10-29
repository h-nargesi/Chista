use RahavardNovin3;
go

--create or alter view Trades as

/*
RESULT_COUNT = 20;
SIGNAL_STEP_COUNT = 40;
SIGNAL_LAST_YEARS = SIGNAL_STEP_COUNT + RESULT_COUNT;
YEARS_COUNT = 3;
*/

---- POINTER
with pointer as (
	select		InstrumentID, RecordType, StartDateEn,
				dbo.jalali(StartDateEn) as StartDateJl
	from (
		select		InstrumentID, RecordType,
					lag(DateTimeEn, 20/*RESULT_COUNT*/)
						over (partition by InstrumentID, RecordTyp order by DateTimeEn desc) as StartDateEn
		from		Trade
		where		RecordType is not null
	) p
	where		StartDateEn is not null

-- RESTRICTION
), ristriction as (
	select		InstrumentID, RecordType, StartDateEn, StartDateJl,
				dateadd(month, 1,
					dbo.jparse(StartDateJl / 10000 - 4, -- YEARS_COUNT + 1
					StartDateJl % 10000 / 100,
					StartDateJl % 100, 0, 0, 0)) as EndDateEn
	from		pointer

-- STREAM
), stream as (
	select		strm.*,
				dbo.jalali(DateTimeEnNext) as DateTimeJlNext,
				floor(DateTimeJl / 10000) as DateTimeJYear
	from (
		select		row_number()
						over (partition by t.InstrumentID, t.RecordType order by t.DateTimeEn desc) as Ranking,
					t.InstrumentID, t.RecordType, r.StartDateEn, r.StartDateJl,
					t.DateTimeEn, dbo.jalali(t.DateTimeEn) as DateTimeJl,
					lead(t.DateTimeEn)
						over (partition by t.InstrumentID, t.RecordType order by DateTimeEn desc) as DateTimeEnNext,
					100 * isnull(t.ClosePriceChange / lead(t.ClosePrice)
						over (partition by t.InstrumentID, t.RecordType order by DateTimeEn desc), 0) as ChangePercent
		from		Trade t, ristriction r
		where		t.InstrumentID = r.InstrumentID and t.RecordType = r.RecordType
				and t.DateTimeEn between r.EndDateEn and r.StartDateEn
	) strm

-- ANNUAL
), annual as (
  select Ranking, InstrumentID, RecordType, DateTimeEn,
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
           annual.DateTimeEn,
           annual.ChangePercent
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
