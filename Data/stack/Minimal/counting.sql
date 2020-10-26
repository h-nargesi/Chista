select company_id, max(count(*) - 100, 0) as amount
from stack
group by company_id