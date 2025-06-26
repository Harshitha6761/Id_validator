"""
Script to generate ID cards with custom photos
"""

import os
from dataset_generator import IDCardDatasetGenerator

def generate_with_custom_photos(photos_dir, output_dir="test_dataset", num_each=50):
    """
    Generate ID cards using custom photos from a directory
    
    Args:
        photos_dir (str): Path to directory containing your photos
        output_dir (str): Output directory for generated ID cards
        num_each (int): Number of ID cards to generate for each category
    """
    
    # Check if photos directory exists
    if not os.path.exists(photos_dir):
        print(f"❌ Photos directory '{photos_dir}' does not exist!")
        print("Please create the directory and add some photos.")
        return None
    
    # Check if directory has photos
    photo_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    photos = []
    for ext in photo_extensions:
        photos.extend([f for f in os.listdir(photos_dir) if f.lower().endswith(ext)])
    
    if not photos:
        print(f"❌ No photos found in '{photos_dir}'!")
        print(f"Supported formats: {', '.join(photo_extensions)}")
        return None
    
    print(f"✅ Found {len(photos)} photos in '{photos_dir}'")
    print(f"📸 Photos: {', '.join(photos[:5])}{'...' if len(photos) > 5 else ''}")
    
    # Create generator with custom photos
    generator = IDCardDatasetGenerator(
        output_dir=output_dir,
        photos_dir=photos_dir
    )
    
    # Generate dataset
    print(f"\n🚀 Generating {num_each} ID cards for each category...")
    dataset = generator.generate_dataset(num_each_category=num_each)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total ID cards: {len(dataset['genuine']) + len(dataset['suspicious']) + len(dataset['fake'])}")
    
    return dataset

def main():
    """Example usage"""
    
    print("🎯 ID Card Generator with Custom Photos")
    print("=" * 50)
    
    # Option 1: Use a specific photos directory
    photos_dir = input("Enter the path to your photos directory (or press Enter to use synthetic photos): ").strip()
    
    if not photos_dir:
        print("\n📝 Using synthetic photos (no custom photos provided)")
        generator = IDCardDatasetGenerator()
        dataset = generator.generate_dataset(num_each_category=50)
    else:
        print(f"\n📸 Using custom photos from: {photos_dir}")
        dataset = generate_with_custom_photos(photos_dir)
    
    if dataset:
        print("\n🎉 Generation complete!")
        print("\n📋 Next steps:")
        print("1. Check the generated ID cards in the test_dataset folder")
        print("2. Run 'python view_photos_simple.py' to see sample results")
        print("3. Train your model with the new dataset")
        print("4. Test the API with the new ID cards")

if __name__ == "__main__":
    main() 