"""Actual local filesystem component checks, not Linux/root/provider proof."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from app.files.transfer import CAPTURE_MAX_BYTES
from app.workspace.media_recipe import SourceInspection, TemplateInventoryEntry, validate_recipe
from app.workspace.media_render import prepare_render
from app.workspace.media_staging import INSTALL_RENDER_INPUTS, expected_materialization_receipt, materialization_request
from tests.test_workspace_media_render import inputs


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='jarvis-media-staging-')
        self.root=Path(self.temporary.name);self.inputs=self.root/'inputs';self.workspace=self.root/'workspace'
        self.inputs.mkdir();self.workspace.mkdir()
        self.program={'__name__':'isolated_component'}
        exec(compile(INSTALL_RENDER_INPUTS,'fixed-media-staging-program','exec'),self.program)
        self.content=b'exact synthetic media bytes\x00'*5000
        digest=hashlib.sha256(self.content).hexdigest();args=inputs()
        value=args['recipe'].recipe.model_dump(mode='json');value['sources'][0]['file']['content_sha256']=digest
        inspection=SourceInspection.model_validate(args['source'].inspection.model_dump() | {
            'file':value['sources'][0]['file'],'size_bytes':len(self.content)})
        args['source']=replace(args['source'],inspection=inspection)
        args['recipe']=validate_recipe(value,inspections=[inspection],policy=args['binding'].policy,
            inventory=[TemplateInventoryEntry(template=value['template'],output_profile=value['output_profile'],ready=True)])
        self.plan=prepare_render(**args);self.request=materialization_request(self.plan)
        self.source=self.inputs/self.request['source_identity'];self.source.write_bytes(self.content);self.source.chmod(0o444)
        self.bundle=self.inputs/('render-'+self.plan.digest)
        self.output=self.workspace/'media'/self.plan.task_id.hex/self.plan.digest

    def tearDown(self):
        # Local test user owns these files; restoring modes is test cleanup only.
        for root,dirs,files in os.walk(self.root,followlinks=False):
            for name in [root,*[str(Path(root)/d) for d in dirs]]:
                if not Path(name).is_symlink():os.chmod(name,0o700)
        self.temporary.cleanup()

    def run_install(self, request=None):
        # Production install() fixes uid0/worker account and roots. Direct fds let
        # this test verify real local bytes/paths without pretending to be root.
        a=os.open(self.inputs,os.O_RDONLY|os.O_DIRECTORY);b=os.open(self.workspace,os.O_RDONLY|os.O_DIRECTORY)
        try:return self.program['materialize'](self.request if request is None else request,a,b,
            privileged_uid=os.getuid(),worker_uid=os.getuid(),worker_gid=os.getgid())
        finally:os.close(b);os.close(a)

    def test_real_bytes_readonly_layout_and_identical_replay_without_overwrite(self):
        result=self.run_install();self.assertEqual(result,expected_materialization_receipt(self.plan))
        source=self.bundle/'staging/inputs'/(self.request['source_sha256']+'.mp4')
        metadata=self.bundle/'input.json'
        self.assertEqual(source.read_bytes(),self.content);self.assertEqual(metadata.read_bytes(),self.plan.envelope_bytes)
        for path in [self.bundle,self.bundle/'staging',source.parent]:self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o555)
        for path in [source,metadata]:self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o444)
        self.assertTrue(self.output.is_dir());self.assertFalse((self.output/'render').exists())
        before={str(p):(p.stat().st_ino,p.stat().st_mtime_ns) for p in [source,metadata]}
        self.assertEqual(self.run_install(),result)
        self.assertEqual(before,{str(p):(p.stat().st_ino,p.stat().st_mtime_ns) for p in [source,metadata]})
        self.assertEqual(self.source.read_bytes(),self.content)

    def test_invalid_request_does_not_create_paths(self):
        for updates in [{'source_identity':'../escape'},{'task_id':'../escape'}, {'extra_command':'touch x'},
                        {'limit_bytes':CAPTURE_MAX_BYTES + 1},{'size_bytes':True},{'schema_version':True},
                        {'envelope_sha256':'0'*64}]:
            with self.subTest(updates=updates),self.assertRaises(ValueError):self.run_install(self.request | updates)
        self.assertFalse(self.bundle.exists());self.assertEqual(list(self.workspace.iterdir()),[])

    def test_wrong_content_hash_and_writable_source_rejected_before_copy(self):
        with self.assertRaises(ValueError):self.run_install(self.request | {'size_bytes':len(self.content)+1})
        self.source.chmod(0o644)
        with self.assertRaises(ValueError):self.run_install()
        self.assertFalse(self.bundle.exists())

    def test_source_symlink_rejected(self):
        other=self.root/'other';self.source.rename(other);self.source.symlink_to(other)
        with self.assertRaises(OSError):self.run_install()
        self.assertFalse(self.bundle.exists());self.assertEqual(other.read_bytes(),self.content)

    def test_bundle_symlink_rejected(self):
        target=self.root/'foreign';target.mkdir();self.bundle.symlink_to(target)
        with self.assertRaises(OSError):self.run_install()
        self.assertEqual(list(target.iterdir()),[])

    def test_worker_parent_symlink_cannot_redirect_output(self):
        target=self.root/'foreign';target.mkdir();(self.workspace/'media').symlink_to(target)
        with self.assertRaises(OSError):self.run_install()
        self.assertEqual(list(target.iterdir()),[])
        self.assertTrue(self.bundle.exists()) # Partial overall outcome is retained.

    def test_partial_bundle_is_preserved_and_never_repaired(self):
        self.bundle.mkdir(mode=0o700);(self.bundle/'partial').write_text('keep evidence')
        with self.assertRaises(ValueError):self.run_install()
        self.assertEqual((self.bundle/'partial').read_text(),'keep evidence')
        self.assertFalse((self.bundle/'input.json').exists())

    def test_changed_existing_envelope_rejected_without_overwrite(self):
        self.run_install();p=self.bundle/'input.json';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
        with self.assertRaises(ValueError):self.run_install()
        self.assertEqual(p.read_bytes(),b'changed')

    def test_root_only_envelope_is_not_mistaken_for_renderer_readable_input(self):
        self.run_install();p=self.bundle/'input.json';p.chmod(0o400)
        with self.assertRaisesRegex(ValueError,'render_input_readability'):self.run_install()
        self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o400) # Reject; never repair silently.

    def test_existing_envelope_symlink_is_not_followed(self):
        self.run_install();self.bundle.chmod(0o755)
        p=self.bundle/'input.json';p.unlink();outside=self.root/'unrelated';outside.write_text('unchanged')
        p.symlink_to(outside);self.bundle.chmod(0o555)
        with self.assertRaises(OSError):self.run_install()
        self.assertEqual(outside.read_text(),'unchanged')

    def test_existing_render_is_preserved_and_requires_original_command_reconciliation(self):
        self.run_install();render=self.output/'render';render.mkdir();result=render/'output.mp4';result.write_bytes(b'prior result')
        with self.assertRaisesRegex(ValueError,'reconcile_original_command'):self.run_install()
        self.assertEqual(result.read_bytes(),b'prior result')

    def test_changed_original_rejected_even_when_bundle_was_prepared(self):
        self.run_install();self.source.chmod(0o644);self.source.write_bytes(b'x'*len(self.content));self.source.chmod(0o444)
        with self.assertRaisesRegex(ValueError,'digest'):self.run_install()
        self.assertEqual((self.bundle/'input.json').read_bytes(),self.plan.envelope_bytes)

    def test_source_change_during_copy_preserves_partial_outcome(self):
        checked=self.program['read_checked']
        def change(fd,**kwargs):
            if kwargs.get('copy_to') is not None:
                self.source.chmod(0o644);self.source.write_bytes(b'x'*len(self.content));self.source.chmod(0o444)
            return checked(fd,**kwargs)
        with patch.dict(self.program,read_checked=change),self.assertRaisesRegex(ValueError,'digest'):self.run_install()
        self.assertTrue(self.bundle.exists());self.assertFalse((self.bundle/'input.json').exists())
        with self.assertRaises(ValueError):self.run_install()

    def test_duplicate_envelope_field_or_unbound_asset_rejected(self):
        for raw in ['{"recipe":{},"sources":[],"sources":[]}',
                    json.dumps({'recipe':{},'sources':[{'assetPath':'../../escape','contentSha256':self.request['source_sha256']}]})]:
            request=self.request | {'envelope':raw,'envelope_sha256':hashlib.sha256(raw.encode()).hexdigest()}
            with self.assertRaises(ValueError):self.run_install(request)
        self.assertFalse(self.bundle.exists())

    def test_source_directory_rejected(self):
        self.source.unlink();self.source.mkdir()
        with self.assertRaises(ValueError):self.run_install()
        self.assertFalse(self.bundle.exists())

    def test_production_entry_does_not_accept_user_supplied_roots_or_unprivileged_caller(self):
        if os.geteuid()==0:self.skipTest('This assertion requires the actual unprivileged local test user')
        with self.assertRaisesRegex(ValueError,'trusted_root_wrapper_required'):
            self.program['install'](self.request)
        self.assertFalse(self.bundle.exists())


if __name__=='__main__':unittest.main()
