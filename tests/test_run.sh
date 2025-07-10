set -euo pipefail

JOB_ID=$(sbatch ~/jgruen/ci-package/ci_${CI_JOB_ID}/tests/pytest.sbatch -w | awk '{print $4}')
LOGFILE="/gpfs/petra3/scratch/jgruen/ci-package/ci-${JOB_ID}-output.log"

echo "Submitted ${JOB_ID}, waiting…"
while :; do
    STATE=$(sacct -j "$JOB_ID" --format=State --noheader | head -n1 | awk '{print $1}')
    [[ "$STATE" == "COMPLETED" || "$STATE" == "FAILED" || "$STATE" == "CANCELLED" ]] && break
    sleep 5
done
echo "Job $JOB_ID ended with state $STATE"
cat /gpfs/petra3/scratch/jgruen/ci-package/ci-${JOB_ID}-output.log
if [[ "$STATE" != "COMPLETED" ]]; then
    echo "❌ Job $JOB_ID failed with state $STATE"
    exit 1
fi
echo "✅ Job $JOB_ID completed successfully."