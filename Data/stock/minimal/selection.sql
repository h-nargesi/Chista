select CompanyID, Ranking, round(100 * (Close - LastClose) / LastClose, 2) as Change
from (
	select CompanyID, Close,
		   lag(Close, 1, Close) over(partition by company_id order by Stack_ID) as LastClose,
		   row_number() over(partition by company_id order by Stack_ID) as Ranking
	from stack
	where company_id = 1
)
order by CompanyID, Ranking
limit {RECORD_COUNT} offset 