SELECT bill_id,
		session_id, 
		introduced_date, 
		final_date, 
		present_date, 
		(final_date - present_date) as "days_to_final", 
		label 
FROM sketch.bill_processed 
ORDER BY present_date