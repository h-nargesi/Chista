use RahavardNovin3;

declare @DateLimit date = dbo.jparse(1388, 1, 1, 0, 0, 0);
declare @LastTradeDate date = dbo.jparse(1399, 5, 1, 0, 0, 0);

/*select	Name,
		(select t.Title from Market t where t.ID = MarketID) as Market,
		(select t.Title from Board t where t.ID = BoardID) as Board,
		(select t.Title from InstrumentGroup t where t.ID = GroupID) as [Group],
		(select t.Title from Exchange t where t.ID = ExchangeID) as Exchange,
		Instrument.*
from	Instrument
where	ID in (
			select ID from Instrument
			where TypeID = 1 and ExchangeID in (1, 2) and MortgageLoanID is null
	)
;*/

update Trade set RecordType = null;

update		Trade
set			RecordType = 'T'
where		DateTimeEn >= @DateLimit
		and	InstrumentID in (
				select		InstrumentID
				from		Trade
				where		DateTimeEn >= @DateLimit
						and InstrumentID in (select ID from Instrument where TypeID = 1 and ExchangeID in (1, 2))
				group by	InstrumentID
				having		max(DateTimeEn) >= @LastTradeDate
						and count(*) >= 810
				order by	count(*) desc offset 0 rows fetch first 300 rows only
		)
;

merge into Trade t
using (
	select top 32 percent
		InstrumentID, Ranking, min(DateTimeEn) as StartDate, max(DateTimeEn) as EndDate,
		case when avg(RecordType) <= 0.5 then 'V' else 'E' end as RecordType
	from (
		select InstrumentID, DateTimeEn, rand(checksum(newid())) RecordType,
			row_number() over (partition by InstrumentID order by DateTimeEn) / 5 as Ranking,
			count(*) over (partition by InstrumentID) / 5 as Maximum
		from Trade
		where RecordType is not null
	) t
	where t.Ranking >= 162 and t.Ranking + 20 < t.Maximum
	group by InstrumentID, Ranking
	order by newid()
) r
on (t.InstrumentID = r.InstrumentID and t.DateTimeEn between r.StartDate and r.EndDate)
when matched then update set t.RecordType = r.RecordType
;

