use RahavardNovin3;

declare @DateLimit date = dbo.jparse(1388, 1, 1, 0, 0, 0);
declare @LastTradeDate date = dbo.jparse(1399, 5, 1, 0, 0, 0);

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

update 	Trade
set 	RecordType = case when rand(checksum(newid())) <= 0.5 then 'V' else 'E' end
where 	RecordType = 'T' and InstrumentID in (
	select		InstrumentID
	from (
		select		InstrumentID,
					sum(Quantity) over (order by newid()) as Stack
		from (
			select 		InstrumentID, count(*) as Quantity
			from 		Trade
			where 		RecordType = 'T'
			group by	InstrumentID
		) inst_q
	) ins_or
	where Stack <= (select count(*) * 0.2 from Trade where RecordType = 'T')
)
;
