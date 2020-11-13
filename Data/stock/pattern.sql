use RahavardNovin3;
go

truncate table pattern;
go

/*
RESULT_COUNT = 20;
SIGNAL_STEP_COUNT = 183;
SIGNAL_LAST_YEARS = 60;
YEARS_COUNT = 3;
*/

with pattern_rst (Ranking) as (
	select	1 as Ranking
	union all
	select	Ranking + 1 as Ranking
	from	pattern_rst
	where	Ranking < (20/*RESULT_COUNT*/)
	
-- PATTERN
), pattern_step (Section, Average, Ranking) as (
	select	1 as Section, 1 as Average, 0 as Ranking
	union all
	select	case when Ranking + 1 < 183/*SIGNAL_STEP_COUNT*/
				then Section
				else Section + 1
			end Section,
			case when Ranking + 1 < 183/*SIGNAL_STEP_COUNT*/
				then Average
				else Average * 2
			end Average,
			case when Ranking + 1 < 183/*SIGNAL_STEP_COUNT*/
				then Ranking + 1
				else 0
			end as Ranking
	from	pattern_step
	where	Ranking + 1 < 183
	--where	Ranking + 1 >= 40/*SIGNAL_STEP_COUNT*/ and (Section + 1) <= 4
		--or	Ranking + 1 < 40/*SIGNAL_STEP_COUNT*/

-- PATTERN
), pattern_year (Section, Ranking) as (
	select	11 as Section, 0 as Ranking
	union all
	select	case when Ranking + 1 < 20/*RESULT_COUNT*/+40/*SIGNAL_STEP_COUNT*/
				then Section
				else Section + 1
			end Section,
			case when Ranking + 1 < 20/*RESULT_COUNT*/+40/*SIGNAL_STEP_COUNT*/
				then Ranking + 1
				else 0
			end as Ranking
	from	pattern_year
	where	Ranking + 1 >= 20/*RESULT_COUNT*/+40/*SIGNAL_STEP_COUNT*/ and (Section + 1) <= 13/*YEARS_COUNT+10*/
		or	Ranking + 1 < 20/*RESULT_COUNT*/+40/*SIGNAL_STEP_COUNT*/
	 
-- PATTERN
), pattern_all (Section, Ranking) as (
	select	0 as Section, Ranking from pattern_rst
	union
	select	Section, (Ranking / Average) + 1 from pattern_step
	union
	select	Section, Ranking / (Section - 10) + 1 from pattern_year
)

insert into pattern
select * from pattern_all
option (maxrecursion 0)
go

select * from pattern;