import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import MAX_NULL_PERCENTAGE, MIN_RECORDS
except ImportError:
    # Fallback values if config import fails
    MAX_NULL_PERCENTAGE = 1.0
    MIN_RECORDS = 50

def perform_quality_check(filepath):  # Renamed to match your DAG
    """Perform data quality checks"""
    
    print(f"\n{'='*60}")
    print(f"Running Data Quality Checks")
    print(f"{'='*60}")
    print(f"File: {filepath}\n")
    
    # Load data
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    quality_issues = []
    
    # Check 1: Minimum number of records
    print(f"✓ Checking minimum records (threshold: {MIN_RECORDS})...")
    if len(df) < MIN_RECORDS:
        quality_issues.append(f"Insufficient records: {len(df)} < {MIN_RECORDS}")
    else:
        print(f"  ✅ PASS: {len(df)} records found")
    
    # Check 2: Null value percentage
    print(f"\n✓ Checking null values (threshold: {MAX_NULL_PERCENTAGE}%)...")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > MAX_NULL_PERCENTAGE:
                quality_issues.append(f"Column '{col}' has {null_pct:.2f}% null values (threshold: {MAX_NULL_PERCENTAGE}%)")
                print(f"  ❌ FAIL: '{col}' has {null_pct:.2f}% nulls")
            else:
                print(f"  ✅ PASS: '{col}' has {null_pct:.2f}% nulls")
    
    # Check 3: Schema validation
    print(f"\n✓ Checking schema...")
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        quality_issues.append(f"Missing required columns: {missing_columns}")
        print(f"  ❌ FAIL: Missing columns {missing_columns}")
    else:
        print(f"  ✅ PASS: All required columns present")
    
    # Check 4: Data type validation
    print(f"\n✓ Checking data types...")
    for col in required_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                quality_issues.append(f"Column '{col}' is not numeric")
                print(f"  ❌ FAIL: '{col}' is not numeric")
            else:
                print(f"  ✅ PASS: '{col}' is numeric")
    
    # Check 5: Logical validation (high >= low, etc.)
    print(f"\n✓ Checking logical constraints...")
    if 'high' in df.columns and 'low' in df.columns:
        invalid_rows = df[df['high'] < df['low']]
        if len(invalid_rows) > 0:
            quality_issues.append(f"Found {len(invalid_rows)} rows where high < low")
            print(f"  ❌ FAIL: {len(invalid_rows)} rows where high < low")
        else:
            print(f"  ✅ PASS: All rows have high >= low")
    
    # Report results
    print(f"\n{'='*60}")
    if quality_issues:
        print("❌ DATA QUALITY CHECK FAILED!")
        print(f"{'='*60}")
        for issue in quality_issues:
            print(f"  ❌ {issue}")
        print(f"{'='*60}\n")
        raise Exception(f"Data quality check failed: {quality_issues}")
    else:
        print("✅ DATA QUALITY CHECK PASSED!")
        print(f"{'='*60}")
        print(f"  📊 Records: {len(df)}")
        print(f"  📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"  ✅ All quality checks passed")
        print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_quality_check.py <filepath>")
        sys.exit(1)
    
    perform_quality_check(sys.argv[1])