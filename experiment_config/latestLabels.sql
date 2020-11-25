select	n1.bill_id,
		n1.session_id, 
		n1.introduced_date, 
		n1.final_date, 
		n1.present_date, 
		n1.days_to_final, 
		n1.days_from_introduction,
		n1.number_dems,
		n1.number_republicans,
		n1.is_bipartisan,
		n1.label,	
		n1.introduced_body,
		n1.subjects,
		n2.primary_sponsor_district,
		n2.ballotpedia
into sketch.bill_processed_district
from (
	select 	m.bill_id,
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
			from (
				select 	bp.bill_id,
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
				from sketch.bill_processed bp join (
												select m.bill_id, m.number_dems, m.number_republicans,
												(
													case  when m.number_dems != 0 and m.number_republicans !=0 then 1
													else 0
													end
												) as "is_bipartisan"
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

	order by m.present_date ) n1 join
							(select distinct on (bs.bill_id)
								bs.bill_id, sp.district as "primary_sponsor_district" , sp.ballotpedia 
								from ml_policy_class.sessions_people sp join ml_policy_class.bill_sponsors bs on sp.person_id = bs.sponsor_id 
									where bs.sponsor_type = 1 and sp.state_id = 32) n2 on n1.bill_id = n2.bill_id
