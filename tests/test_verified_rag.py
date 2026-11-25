import unittest, tempfile, shutil, json
from pathlib import Path
from types import SimpleNamespace
from Fronted import verified_rag as v
from Fronted.force_rules import _force_include_articles
from Fronted.prompt_helpers import finalize_reply, OUT_OF_SCOPE_TEXT
from Fronted.labor_calculator import calculate_overtime

class FakeClient:
    def __init__(self, body, stop='end_turn'):
        self.calls=0; self.body=body; self.stop=stop; self.messages=self
    def create(self, **kwargs):
        self.calls+=1
        return SimpleNamespace(stop_reason=self.stop,content=[SimpleNamespace(type='text',text=self.body)])

def selection(key='N0030001:24',lines=None):
    return json.dumps({'status':'ok','articles':[{'id':key,'lines':lines if lines is not None else [1]}]})

def official(code):
    return {a.no:a.body for a in v.catalog().values() if a.code==code},'2026-11-25T21:00:00+08:00'

class EvidenceTests(unittest.TestCase):
    def test_catalog_raw_and_cleaned(self):
        a=v.load_catalog(); self.assertEqual(len(a),161);self.assertIn('N0030001:24',a);self.assertIn('N0030002:24',a)
    def test_raw_changed(self):
        with tempfile.TemporaryDirectory() as t:
            for _,stem in v.FILES.values():
                for suffix in ('.json','_cleaned.json'):shutil.copy(v.DATA/(stem+suffix),t)
            p=Path(t)/'labor_law.json';d=json.loads(p.read_text());d[0]['content']='tampered';p.write_text(json.dumps(d));
            with self.assertRaises(ValueError):v.load_catalog(t)
    def test_exact_no_model(self):
        c=FakeClient('not called');out=v.answer('勞基法第24條',client=c,online=False);self.assertEqual(c.calls,0);self.assertIn('勞動基準法第24條',out);self.assertIn('未確認現行版本',out)
    def test_exact_rules_identity(self):
        out=v.answer('勞動基準法施行細則第24條',online=False); self.assertIn('N0030002&flno=24',out); self.assertNotIn('N0030001&flno=24',out)
    def test_exact_subarticle(self):self.assertEqual(v.direct_ids('勞基法第32條之1'),['N0030001:32-1'])
    def test_deleted(self):self.assertEqual(v.answer('勞動基準法施行細則第4-1條',online=False),v.REFUSAL)
    def test_nonexistent(self):self.assertEqual(v.answer('勞基法第999條',online=False),v.REFUSAL)
    def test_not_exact_scenario(self):self.assertIsNone(v.direct_ids('依勞基法第24條，公司欠我多少錢？'))
    def test_faq_no_model(self):
        c=FakeClient('invalid');out=v.answer('加班費怎麼算？',client=c,online=False);self.assertEqual(c.calls,0);self.assertIn('第24條',out);self.assertIn('第32條',out)
    def test_bm25_forced(self):self.assertTrue({'N0030001:24','N0030001:32','N0030001:36'} <= {a.key for a in v.retrieve('休息日加班費如何計算')})
    def test_model_one_call(self):
        c=FakeClient(selection());out=v.answer('加班四小時公司怎麼給錢',client=c,online=True,fetch=official);self.assertEqual(c.calls,1);self.assertIn('官方網頁原文與本地資料一致',out);self.assertIn(v.catalog()['N0030001:24'].lines[0],out)
    def test_model_unknown_id(self):self.assertEqual(v.answer('加班四小時公司怎麼給錢',client=FakeClient(selection('N0030001:999')),online=False),v.REFUSAL)
    def test_model_fake_quote(self):self.assertEqual(v.answer('加班四小時公司怎麼給錢',client=FakeClient(selection(lines=['假的法條'])),online=False),v.REFUSAL)
    def test_model_incomplete(self):self.assertEqual(v.answer('加班四小時公司怎麼給錢',client=FakeClient(selection(),stop='max_tokens'),online=False),v.REFUSAL)
    def test_model_free_prose(self):self.assertEqual(v.answer('加班四小時公司怎麼給錢',client=FakeClient('公司一定違法'),online=False),v.REFUSAL)
    def test_model_out(self):self.assertEqual(v.answer('美國加班費法律',client=FakeClient('{"status":"out","articles":[]}'),online=False),v.OUT)
    def test_missing_key(self):self.assertEqual(v.answer('公司違法嗎',online=False),v.REFUSAL)
    def test_outage(self):
        def fail(_):raise TimeoutError()
        self.assertIn('不得當成現行規定',v.answer('勞基法第24條',online=True,fetch=fail))
    def test_changed_official(self):
        out=v.answer('勞基法第24條',online=True,fetch=lambda _:({'24':'新資料'},'now'));self.assertIn('已停止輸出舊條文',out);self.assertNotIn(v.catalog()['N0030001:24'].lines[0],out)
    def test_official_parser(self):
        html=''.join(f'<div class="row no-gutters"><div class="col-no">第 {i} 條</div><div class="col-data">測試{i}</div></div>' for i in range(1,12));self.assertEqual(v.parse_official(html)['1'],'測試1')
        with self.assertRaises(ValueError):v.parse_official('<html>錯誤</html>')
    def test_force_source_and_dedup(self):
        r=[('a',{'source':v.RULES,'article_no':'第 24 條'},1)];m={'source':v.ACT,'article_no':'第 24 條'};out=_force_include_articles('加班',r,['b','c'],[m,m],['加班'],{'第 24 條'});self.assertEqual(len(out),2);self.assertEqual(out[0][1]['source'],v.ACT)
    def test_old_parser_fails_closed(self):self.assertEqual(finalize_reply('不明答案'),OUT_OF_SCOPE_TEXT)
    def test_case_metadata(self):
        rows=v.search_cases('加班費');self.assertGreater(len(rows),0);self.assertTrue(all(m.get('url') for _,m,_ in rows)); self.assertTrue(all(m.get('content')==text for text,m,_ in rows))
    def test_weekday_over_limit(self):
        with self.assertRaises(ValueError):calculate_overtime(36000,5,'平日')
    def test_nonfinite(self):
        for x in (float('nan'),float('inf'),-1,0,True):
            with self.assertRaises(ValueError):calculate_overtime(x,2,'平日')
    def test_fraction(self):self.assertAlmostEqual(calculate_overtime(36000,.5,'平日')['total'],100)
    def test_rest_12(self):self.assertAlmostEqual(calculate_overtime(36000,12,'休息日')['total'],3500)

if __name__=='__main__':unittest.main()
