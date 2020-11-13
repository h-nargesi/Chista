use RahavardNovin3;

select		InstrumentID, count(*) as Amount,
			case max(RecordType)
				when 'T' then 'Training'
				when 'V' then 'Validation'
				when 'E' then 'Evaluation'
				else 'Unkown'
			end as RecordType
from		Trade
where		RecordType is not null
group by	InstrumentID
order by	InstrumentID-- desc
;

select		count(*) as Amount,
			sum(iif(RecordType = 'T', 1, 0)) as Training,
			sum(iif(RecordType = 'T', 100.0, 0)) / count(*) as Training,
			sum(iif(RecordType = 'V', 1, 0)) as Validation,
			sum(iif(RecordType = 'V', 100.0, 0)) / count(*) as Validation,
			sum(iif(RecordType = 'E', 1, 0)) as Evaluation,
			sum(iif(RecordType = 'E', 100.0, 0)) / count(*) as Evaluation
from		Trade
where		RecordType is not null
order by	Amount desc
;