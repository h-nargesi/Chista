use RahavardNovin3;

select		InstrumentID, count(*) as Amount,
			sum(iif(RecordType = 'X', 1, 0)) as Available,
			sum(iif(RecordType = 'X', 100.0, 0)) / count(*) as Exercise,
			sum(iif(RecordType = 'V', 100.0, 0)) / count(*) as Validation,
			sum(iif(RecordType = 'T', 100.0, 0)) / count(*) as Test
from		Trade
where		RecordType is not null
group by	InstrumentID
order by	InstrumentID-- desc
;

select		count(*) as Amount,
			sum(iif(RecordType = 'X', 1, 0)) as Available,
			sum(iif(RecordType = 'X', 100.0, 0)) / count(*) as Exercise,
			sum(iif(RecordType = 'V', 100.0, 0)) / count(*) as Validation,
			sum(iif(RecordType = 'T', 100.0, 0)) / count(*) as Test
from		Trade
where		RecordType is not null
order by	Amount desc
;