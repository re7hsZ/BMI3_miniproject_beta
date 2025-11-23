"""
Minimal test to isolate the ValueError
"""
import sys
import traceback

try:
    # Test 1: Import simulator
    print("Test 1: Importing simulator...")
    from src.simulator import generate_dataset
    print("✓ Import successful")
    
    # Test 2: Generate minimal dataset
    print("\nTest 2: Generating 5 host genes...")
    data = generate_dataset(5, 0, 0.4, 0.0)
    print(f"✓ Generated {len(data)} items")
    print(f"  Sample item: {data[0]}")
    print(f"  Item structure: {len(data[0])} elements")
    
    # Test 3: Import Bio
    print("\nTest 3: Importing Bio...")
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO
    print("✓ Bio imports successful")
    
    # Test 4: Create SeqRecord
    print("\nTest 4: Creating SeqRecord from data...")
    gid, seq, label = data[0]
    record = SeqRecord(Seq(seq), id=gid, description="")
    print(f"✓ SeqRecord created: {record.id}")
    
    # Test 5: Save to file
    print("\nTest 5: Saving to FASTA...")
    records = [SeqRecord(Seq(item[1]), id=item[0], description="") for item in data]
    SeqIO.write(records, "test_output.fasta", "fasta")
    print(f"✓ Saved {len(records)} records")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
