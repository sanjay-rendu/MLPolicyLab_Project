select m.bill_id,
					m.session_id, 
					m.introduced_date, 
					m.final_date, 
					m.present_date, 
					(m.final_date - m.present_date) as "days_to_final", 
					(m.present_date- m.introduced_date) as "days_from_introduction",
					m.number_dems,
					m.number_republicans,
					m.is_bipartisan,
					m.label,	
					bills.introduced_body,
					bills.subjects
	from (select bp.bill_id,
					bp.session_id, 
					bp.introduced_date, 
					bp.final_date, 
					bp.present_date, 
					(bp.final_date - bp.present_date) as "days_to_final", 
					(bp.present_date- bp.introduced_date) as "days_from_introduction",
					rc.number_dems,
					rc.number_republicans,
					rc.is_bipartisan,
					bp.label
			from sketch.bill_processed bp join (select m.bill_id, m.number_dems, m.number_republicans,
												(case  when m.number_dems != 0 and m.number_republicans !=0 then 1
												else 0
												end) as "is_bipartisan"
												from (select b.bill_id, sum(b.democrats) as "number_dems", sum(b.republicans) as "number_republicans"
					
												from (select bill_id, (case  when party_id = 1 then 1
																		else 0
																		end) as "democrats", 
																		(case when party_id = 2 then 1
																		else 0
																		end) as "republicans"
														from ml_policy_class.bill_sponsors) b
												group by b.bill_id) m) rc on rc.bill_id = bp.bill_id) m join ml_policy_class.bills bills 
												on bills.bill_id = m.bill_id

order by m.present_date 