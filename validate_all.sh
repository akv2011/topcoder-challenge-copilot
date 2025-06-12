#!/bin/bash

# AI Copilot Agent - Master Validation Script
# This script runs all validation checks to ensure deliverables are ready

echo "🎯 AI COPILOT AGENT - MASTER VALIDATION SUITE"
echo "=============================================="
echo "📅 Date: $(date)"
echo "=============================================="

# Make all Python scripts executable
chmod +x validate_deliverables.py
chmod +x test_functionality.py  
chmod +x final_checklist.py
chmod +x run_complete_validation.py

echo ""
echo "🔍 Running Complete Validation Suite..."
echo "--------------------------------------"

# Run the comprehensive validation
python3 run_complete_validation.py

VALIDATION_EXIT_CODE=$?

echo ""
echo "🧪 Running Pre-Submission Tests..."
echo "---------------------------------"

# Run submission readiness tests
python3 test_submission_ready.py

SUBMISSION_EXIT_CODE=$?

echo ""
echo "📋 Running Detailed Checklist..."
echo "-------------------------------"

# Run detailed checklist
python3 final_checklist.py

echo ""
echo "🔧 Running Technical Functionality Tests..."
echo "------------------------------------------"

# Run technical tests
python3 test_functionality.py

echo ""
echo "📊 Running Deliverables Validation..."
echo "------------------------------------"

# Run deliverables check
python3 validate_deliverables.py

echo ""
echo "=============================================="
echo "🏆 MASTER VALIDATION COMPLETE"
echo "=============================================="

# Check results
if [ $VALIDATION_EXIT_CODE -eq 0 ] && [ $SUBMISSION_EXIT_CODE -eq 0 ]; then
    echo "✅ ALL VALIDATIONS PASSED"
    echo "🚀 PROJECT IS READY FOR SUBMISSION!"
    echo ""
    echo "📋 Key Files Generated:"
    echo "  • SUBMISSION_SUMMARY.json - Complete project assessment"
    echo "  • submission_test_results.json - Submission readiness results"
    echo "  • validation_report.json - Detailed validation results"
    echo "  • technical_report.json - Technical implementation analysis"
    echo "  • deliverables_checklist.json - Final checklist results"
else
    echo "⚠️ SOME ISSUES FOUND - CHECK REPORTS FOR DETAILS"
    echo "📋 Generated Reports:"
    echo "  • SUBMISSION_SUMMARY.json - Issues summary"
    echo "  • validation_report.json - Detailed findings"
    echo "  • submission_test_results.json - Test results"
fi

echo ""
echo "🎯 Next Steps:"
if [ $VALIDATION_EXIT_CODE -eq 0 ] && [ $SUBMISSION_EXIT_CODE -eq 0 ]; then
    echo "  1. Review SUBMISSION_SUMMARY.json"
    echo "  2. Run manual tests: See TESTING_GUIDE.md"
    echo "  3. Start backend: cd backend && python main.py"
    echo "  4. Start frontend: cd frontend && npm install && npm run dev"
    echo "  5. Test complete user workflow"
    echo "  6. Submit to Topcoder!"
else
    echo "  1. Review validation reports"
    echo "  2. Fix identified issues"
    echo "  3. Re-run validation"
    echo "  4. Use TESTING_GUIDE.md for manual testing"
fi

exit $VALIDATION_EXIT_CODE
