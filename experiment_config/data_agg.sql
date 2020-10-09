SELECT * 
FROM   (SELECT bp.bill_id, 
               bp.final_status, 
               s.session_id, 
               s.state_id, 
               s.special, 
               s.year_start, 
               s.year_end, 
               b.bill_type, 
               b.subjects, 
               b.introduced_date, 
               b.introduced_body, 
               b.url 
        FROM   (SELECT DISTINCT m.bill_id      AS bill_id, 
                                m.final_status AS final_status 
                FROM   (SELECT bill_id, 
                               ( CASE 
                                   WHEN bill_status = 4 THEN 1 
                                   ELSE 0 
                                 END ) AS final_status 
                        FROM   ml_policy_class.bill_progress) m) bp 
               JOIN ml_policy_class.bills b 
                 ON b.bill_id = bp.bill_id 
               JOIN ml_policy_class.sessions s 
                 ON s.session_id = b.session_id 
        WHERE  s.state_id = 32) bill_details 
       JOIN ml_policy_class.bill_sponsors bs 
         ON bill_details.bill_id = bs.bill_id 