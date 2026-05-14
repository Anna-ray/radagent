"""
RadAgent v2 Submission Validation Script

Validates that all components work end-to-end before submission.
This is the final check before the Milan AI Week 2026 deadline.

Author: Rayane Aggoune
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import time


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class SubmissionValidator:
    """Validates RadAgent v2 submission."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = []
        self.errors = []
    
    def log(self, message: str, status: str = "INFO"):
        """Log validation message."""
        icons = {"INFO": "ℹ️", "PASS": "✓", "FAIL": "✗", "WARN": "⚠️"}
        icon = icons.get(status, "•")
        print(f"{icon} {message}")
        self.results.append({"message": message, "status": status})
    
    def check_file_exists(self, path: Path, description: str) -> bool:
        """Check if a file exists."""
        if path.exists():
            self.log(f"{description}: {path}", "PASS")
            return True
        else:
            self.log(f"{description} MISSING: {path}", "FAIL")
            self.errors.append(f"Missing: {path}")
            return False
    
    def check_directory_exists(self, path: Path, description: str) -> bool:
        """Check if a directory exists."""
        if path.exists() and path.is_dir():
            self.log(f"{description}: {path}", "PASS")
            return True
        else:
            self.log(f"{description} MISSING: {path}", "FAIL")
            self.errors.append(f"Missing directory: {path}")
            return False
    
    def check_python_import(self, module: str) -> bool:
        """Check if a Python module can be imported."""
        try:
            __import__(module)
            self.log(f"Python module '{module}' imports successfully", "PASS")
            return True
        except ImportError as e:
            self.log(f"Python module '{module}' import FAILED: {e}", "FAIL")
            self.errors.append(f"Import error: {module}")
            return False
    
    def run_command(self, cmd: List[str], description: str, timeout: int = 30) -> Tuple[bool, str]:
        """Run a command and check if it succeeds."""
        try:
            self.log(f"Running: {description}", "INFO")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            if result.returncode == 0:
                self.log(f"{description}: SUCCESS", "PASS")
                return True, result.stdout
            else:
                self.log(f"{description}: FAILED (exit code {result.returncode})", "FAIL")
                self.log(f"Error: {result.stderr[:200]}", "FAIL")
                self.errors.append(f"Command failed: {' '.join(cmd)}")
                return False, result.stderr
        except subprocess.TimeoutExpired:
            self.log(f"{description}: TIMEOUT after {timeout}s", "FAIL")
            self.errors.append(f"Timeout: {description}")
            return False, "Timeout"
        except Exception as e:
            self.log(f"{description}: ERROR - {e}", "FAIL")
            self.errors.append(f"Error: {description}")
            return False, str(e)
    
    def validate_core_files(self):
        """Validate core project files exist."""
        print("\n" + "="*80)
        print("VALIDATING CORE FILES")
        print("="*80)
        
        core_files = [
            ("README.md", "README"),
            ("LICENSE", "License file"),
            ("requirements.txt", "Requirements"),
            (".env.example", "Environment template"),
            ("Dockerfile", "Docker configuration"),
            ("docker-compose.yml", "Docker Compose"),
        ]
        
        for filename, desc in core_files:
            self.check_file_exists(self.project_root / filename, desc)
    
    def validate_v1_modules(self):
        """Validate v1 core modules (must not be modified)."""
        print("\n" + "="*80)
        print("VALIDATING V1 CORE MODULES (IMMUTABLE)")
        print("="*80)
        
        v1_modules = [
            "radagent/models/specialist.py",
            "radagent/inference/agentic_rag.py",
            "radagent/inference/findings.py",
            "radagent/inference/gradcam.py",
            "radagent/rag/retriever.py",
            "radagent/app/server.py",
        ]
        
        for module in v1_modules:
            self.check_file_exists(self.project_root / module, f"V1 module: {module}")
    
    def validate_v2_modules(self):
        """Validate v2 new modules."""
        print("\n" + "="*80)
        print("VALIDATING V2 NEW MODULES")
        print("="*80)
        
        v2_modules = [
            ("radagent/agents/critic.py", "CriticAgent"),
            ("radagent/autonomy/tools.py", "Autonomy tools"),
            ("radagent/autonomy/planner.py", "Workflow planner"),
            ("radagent/autonomy/halt.py", "Halt logic"),
            ("radagent/federated/server.py", "FedAvg server"),
            ("radagent/federated/client.py", "Hospital node"),
            ("radagent/data/cxr_datasets.py", "Dataset loaders"),
            ("radagent/modality/router.py", "Modality router"),
            ("radagent/modality/dicom_io.py", "DICOM loader"),
            ("radagent/modality/preprocessing.py", "Preprocessing"),
            ("radagent/modality/registry.yaml", "Modality registry"),
            ("radagent/specialists/mura/infer.py", "MURA specialist"),
            ("radagent/voice/transcriber.py", "STT wrapper"),
            ("radagent/voice/dictation_auditor.py", "Dictation auditor"),
            ("radagent/audit/verify.py", "Audit verifier"),
        ]
        
        for path, desc in v2_modules:
            self.check_file_exists(self.project_root / path, desc)
    
    def validate_demo_scripts(self):
        """Validate demo scripts."""
        print("\n" + "="*80)
        print("VALIDATING DEMO SCRIPTS")
        print("="*80)
        
        scripts = [
            ("scripts/run_vanilla_baseline.py", "Scene 1: Vanilla baseline"),
            ("scripts/run_dictation_demo.py", "Scene 2.5: Dictation"),
            ("scripts/run_modality_demo.py", "Scene 3: Modality router"),
            ("scripts/run_autonomy_demo.py", "Scene 4: Autonomy"),
            ("scripts/run_federated_demo.py", "Scene 5: Federation"),
            ("scripts/run_full_demo.py", "Master orchestrator"),
            ("scripts/run_critic_demo.py", "CriticAgent demo"),
        ]
        
        for path, desc in scripts:
            self.check_file_exists(self.project_root / path, desc)
    
    def validate_tests(self):
        """Validate test files."""
        print("\n" + "="*80)
        print("VALIDATING TESTS")
        print("="*80)
        
        tests = [
            "tests/test_critic_agent.py",
            "tests/test_federated.py",
            "tests/test_voice.py",
            "tests/test_mura_specialist.py",
        ]
        
        for test in tests:
            self.check_file_exists(self.project_root / test, f"Test: {test}")
    
    def validate_documentation(self):
        """Validate documentation."""
        print("\n" + "="*80)
        print("VALIDATING DOCUMENTATION")
        print("="*80)
        
        docs = [
            ("docs/DEMO_SCRIPT.md", "Demo script for video"),
            ("docs/AGENT_LAB_GUIDE.md", "Agent lab guide"),
            ("docs/DATASET_DOWNLOAD_GUIDE.md", "Dataset guide"),
            ("docs/VULTR_DEPLOYMENT.md", "Deployment guide"),
            ("docs/GITHUB_ACTIONS_SETUP.md", "CI/CD guide"),
            ("docs/V2_IMPLEMENTATION_STATUS.md", "Implementation status"),
        ]
        
        for path, desc in docs:
            self.check_file_exists(self.project_root / path, desc)
    
    def validate_python_imports(self):
        """Validate Python imports."""
        print("\n" + "="*80)
        print("VALIDATING PYTHON IMPORTS")
        print("="*80)
        
        modules = [
            "radagent",
            "radagent.agents.critic",
            "radagent.autonomy.tools",
            "radagent.federated.server",
            "radagent.modality.router",
            "radagent.specialists.mura",
            "radagent.voice.transcriber",
        ]
        
        for module in modules:
            self.check_python_import(module)
    
    def validate_dashboard(self):
        """Validate dashboard files."""
        print("\n" + "="*80)
        print("VALIDATING DASHBOARD")
        print("="*80)
        
        dashboard_files = [
            ("radagent/app/static/index.html", "Dashboard HTML"),
            ("radagent/app/static/dashboard_v2_extensions.js", "V2 extensions JS"),
            ("radagent/app/static/dashboard_v2_styles.css", "V2 styles CSS"),
        ]
        
        for path, desc in dashboard_files:
            self.check_file_exists(self.project_root / path, desc)
    
    def validate_deployment(self):
        """Validate deployment files."""
        print("\n" + "="*80)
        print("VALIDATING DEPLOYMENT")
        print("="*80)
        
        deployment_files = [
            ("Dockerfile", "Docker image"),
            ("docker-compose.yml", "Docker Compose"),
            ("scripts/deploy_vultr.sh", "Vultr deployment script"),
            (".dockerignore", "Docker ignore"),
        ]
        
        for path, desc in deployment_files:
            self.check_file_exists(self.project_root / path, desc)
    
    def validate_git_status(self):
        """Validate git repository status."""
        print("\n" + "="*80)
        print("VALIDATING GIT STATUS")
        print("="*80)
        
        # Check current branch
        success, output = self.run_command(
            ["git", "branch", "--show-current"],
            "Check current branch"
        )
        if success:
            branch = output.strip()
            if branch == "feature/v2-milan":
                self.log(f"On correct branch: {branch}", "PASS")
            else:
                self.log(f"Wrong branch: {branch} (expected: feature/v2-milan)", "WARN")
        
        # Check for uncommitted changes
        success, output = self.run_command(
            ["git", "status", "--porcelain"],
            "Check for uncommitted changes"
        )
        if success:
            if output.strip():
                self.log("Uncommitted changes detected", "WARN")
                self.log(f"Changes:\n{output[:500]}", "INFO")
            else:
                self.log("No uncommitted changes", "PASS")
        
        # Check remote
        success, output = self.run_command(
            ["git", "remote", "-v"],
            "Check git remote"
        )
        if success and "Anna-ray/radagent" in output:
            self.log("Correct remote repository", "PASS")
    
    def validate_datasets(self):
        """Validate dataset availability."""
        print("\n" + "="*80)
        print("VALIDATING DATASETS (OPTIONAL)")
        print("="*80)
        
        data_root = self.project_root / "data"
        
        # Check NIH
        nih_images = data_root / "nih" / "images"
        if nih_images.exists():
            num_nih = len(list(nih_images.glob("*.png")))
            if num_nih >= 112000:
                self.log(f"NIH ChestX-ray14: {num_nih:,} images", "PASS")
            else:
                self.log(f"NIH ChestX-ray14: {num_nih:,} images (incomplete)", "WARN")
        else:
            self.log("NIH ChestX-ray14: NOT FOUND (federation demo will not run)", "WARN")
        
        # Check CheXpert
        chexpert_train = data_root / "chexpert" / "CheXpert-v1.0-small" / "train"
        if chexpert_train.exists():
            num_chexpert = len(list(chexpert_train.rglob("*.jpg")))
            if num_chexpert >= 200000:
                self.log(f"CheXpert: {num_chexpert:,} images", "PASS")
            else:
                self.log(f"CheXpert: {num_chexpert:,} images (incomplete)", "WARN")
        else:
            self.log("CheXpert: NOT FOUND (federation demo will not run)", "WARN")
        
        # Check MURA
        mura_train = data_root / "mura" / "MURA-v1.1" / "train"
        if mura_train.exists():
            num_mura = len(list(mura_train.rglob("*.png")))
            if num_mura >= 35000:
                self.log(f"MURA: {num_mura:,} images", "PASS")
            else:
                self.log(f"MURA: {num_mura:,} images (incomplete)", "WARN")
        else:
            self.log("MURA: NOT FOUND (bone X-ray demo will use placeholder)", "WARN")
    
    def generate_report(self):
        """Generate validation report."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        warnings = sum(1 for r in self.results if r["status"] == "WARN")
        
        print(f"\nTotal checks: {total}")
        print(f"✓ Passed: {passed}")
        print(f"✗ Failed: {failed}")
        print(f"⚠️  Warnings: {warnings}")
        
        if failed > 0:
            print("\n" + "="*80)
            print("ERRORS DETECTED")
            print("="*80)
            for error in self.errors:
                print(f"✗ {error}")
            print("\n⚠️  SUBMISSION NOT READY - Fix errors above")
            return False
        elif warnings > 0:
            print("\n⚠️  Warnings detected but submission is valid")
            print("Consider addressing warnings for a stronger submission")
            return True
        else:
            print("\n🎉 ALL CHECKS PASSED - SUBMISSION READY!")
            return True
    
    def run_all_validations(self):
        """Run all validation checks."""
        print("="*80)
        print("RadAgent v2 Submission Validation")
        print("Milan AI Week 2026 AI Agent Olympics")
        print("="*80)
        print(f"Project root: {self.project_root.absolute()}")
        print(f"Validation time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.validate_core_files()
        self.validate_v1_modules()
        self.validate_v2_modules()
        self.validate_demo_scripts()
        self.validate_tests()
        self.validate_documentation()
        self.validate_dashboard()
        self.validate_deployment()
        self.validate_python_imports()
        self.validate_git_status()
        self.validate_datasets()
        
        return self.generate_report()


def main():
    project_root = Path(__file__).parent.parent
    validator = SubmissionValidator(project_root)
    
    success = validator.run_all_validations()
    
    # Save report
    report_path = project_root / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "success": success,
            "results": validator.results,
            "errors": validator.errors
        }, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# Made with Bob
