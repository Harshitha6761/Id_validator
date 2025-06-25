"""
Dataset Generator for College ID Validator Testing
Creates synthetic ID cards for genuine, suspicious, and fake categories
"""

import os
import json
import base64
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import io
from typing import List, Dict, Any
import numpy as np

class IDCardDatasetGenerator:
    def __init__(self, output_dir="test_dataset"):
        self.output_dir = output_dir
        self.create_directories()
        
        # Sample data for generating realistic IDs
        self.colleges = [
            "IIT Bombay", "NIT Warangal", "JNTU Hyderabad", "IIT Delhi", 
            "IIT Madras", "BITS Pilani", "VIT University", "SRM University",
            "Anna University", "Osmania University"
        ]
        
        self.fake_colleges = [
            "MIT India", "Harvard College India", "Stanford University Delhi",
            "Cambridge Institute", "Oxford College Mumbai", "Yale University Pune"
        ]
        
        self.courses = [
            "B.Tech Computer Science", "B.Tech Electronics", "B.Tech Mechanical",
            "M.Tech Software Engineering", "B.Sc Computer Science", "M.Sc Data Science",
            "MBA", "B.Com", "M.Com", "BCA", "MCA"
        ]
        
        self.names = [
            "Arjun Sharma", "Priya Patel", "Rohit Kumar", "Sneha Singh", "Vikram Reddy",
            "Ananya Gupta", "Karthik Nair", "Meera Joshi", "Aditya Verma", "Kavya Rao",
            "Suresh Iyer", "Pooja Agarwal", "Nikhil Desai", "Riya Malhotra", "Akash Pandey"
        ]
        
        self.years = ["2021", "2022", "2023", "2024", "2025"]
        
    def create_directories(self):
        """Create directory structure for dataset"""
        categories = ["genuine", "suspicious", "fake"]
        for category in categories:
            os.makedirs(os.path.join(self.output_dir, category), exist_ok=True)
        
        os.makedirs(os.path.join(self.output_dir, "metadata"), exist_ok=True)
    
    def get_default_font(self, size=20):
        """Get a default font or create a simple one"""
        try:
            # Try to load a system font
            return ImageFont.truetype("arial.ttf", size)
        except:
            try:
                return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", size)
            except:
                try:
                    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
                except:
                    return ImageFont.load_default()
    
    def generate_genuine_id(self, index: int) -> Dict[str, Any]:
        """Generate a genuine-looking ID card"""
        # Standard ID card dimensions
        width, height = 400, 250
        
        # Professional color scheme
        bg_color = (255, 255, 255)  # White background
        header_color = (41, 128, 185)  # Professional blue
        text_color = (0, 0, 0)
        
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Get fonts
        title_font = self.get_default_font(16)
        header_font = self.get_default_font(14)
        text_font = self.get_default_font(11)
        
        # Sample data
        college = random.choice(self.colleges)
        name = random.choice(self.names)
        roll_no = f"{random.randint(10, 99)}{random.choice(['CS', 'EC', 'ME', 'EE'])}{random.randint(1000, 9999)}"
        course = random.choice(self.courses)
        year = random.choice(self.years)
        
        # Header section
        draw.rectangle([0, 0, width, 40], fill=header_color)
        draw.text((width//2 - 80, 10), "STUDENT ID CARD", fill=(255, 255, 255), font=title_font)
        
        # College name
        draw.text((20, 55), college, fill=header_color, font=header_font)
        
        # Photo placeholder (realistic position)
        photo_x, photo_y = 20, 80
        photo_size = 70
        draw.rectangle([photo_x, photo_y, photo_x + photo_size, photo_y + photo_size], 
                      outline=(200, 200, 200), width=2)
        draw.text((photo_x + 15, photo_y + 25), "PHOTO", fill=(150, 150, 150), font=text_font)
        
        # Student details
        details_x = photo_x + photo_size + 20
        y_offset = 85
        
        details = [
            f"Name: {name}",
            f"Roll No: {roll_no}",
            f"Course: {course}",
            f"Year: {year}",
            f"Valid Till: 2025-06-30"
        ]
        
        for detail in details:
            draw.text((details_x, y_offset), detail, fill=text_color, font=text_font)
            y_offset += 18
        
        # Add some official elements
        draw.rectangle([20, height-30, width-20, height-10], outline=(100, 100, 100), width=1)
        draw.text((25, height-27), "Authorized Student Identity Card", fill=(100, 100, 100), font=text_font)
        
        # Add subtle logo area
        draw.ellipse([width-60, 55, width-20, 95], outline=header_color, width=2)
        draw.text((width-50, 70), "LOGO", fill=header_color, font=text_font)
        
        metadata = {
            "category": "genuine",
            "college": college,
            "name": name,
            "roll_no": roll_no,
            "course": course,
            "year": year,
            "expected_label": "genuine",
            "expected_status": "approved"
        }
        
        return {"image": image, "metadata": metadata, "filename": f"genuine_id_{index:03d}.jpg"}
    
    def generate_suspicious_id(self, index: int) -> Dict[str, Any]:
        """Generate a suspicious ID card (poor quality, cropped, etc.)"""
        width, height = random.choice([(200, 125), (150, 100), (300, 180)])  # Unusual sizes
        
        # Poor color choices
        bg_color = (240, 240, 240)
        text_color = (50, 50, 50)  # Low contrast
        
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        font = self.get_default_font(max(8, width//30))
        
        college = random.choice(self.colleges)
        name = random.choice(self.names)
        roll_no = f"{random.randint(10, 99)}{random.choice(['CS', 'EC'])}{random.randint(100, 999)}"  # Shorter roll no
        
        # Cramped layout
        y_offset = 10
        texts = [
            "ID CARD",
            college[:15] + "..." if len(college) > 15 else college,  # Truncated
            f"Name: {name[:10]}...",  # Truncated name
            f"Roll: {roll_no}",
            "Year: 2024"
        ]
        
        for text in texts:
            draw.text((10, y_offset), text, fill=text_color, font=font)
            y_offset += max(12, height//15)
        
        # Add some quality issues
        # 1. Add noise
        noise_factor = 0.1
        img_array = np.array(image)
        noise = np.random.normal(0, 25, img_array.shape)
        noisy_array = np.clip(img_array + noise * noise_factor, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_array)
        
        # 2. Blur slightly
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 3. Reduce contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.7)
        
        metadata = {
            "category": "suspicious",
            "college": college,
            "name": name,
            "roll_no": roll_no,
            "issues": ["poor_quality", "unusual_dimensions", "low_contrast"],
            "expected_label": "suspicious",
            "expected_status": "manual_review"
        }
        
        return {"image": image, "metadata": metadata, "filename": f"suspicious_id_{index:03d}.jpg"}
    
    def generate_fake_id(self, index: int) -> Dict[str, Any]:
        """Generate a fake ID card"""
        fake_type = random.choice(["template_mismatch", "fake_college", "poor_forgery", "non_id"])
        
        if fake_type == "non_id":
            return self.generate_non_id_image(index)
        
        width, height = 400, 250
        
        # Unprofessional colors
        bg_colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 150)]
        bg_color = random.choice(bg_colors)
        
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        font = self.get_default_font(12)
        title_font = self.get_default_font(14)
        
        if fake_type == "fake_college":
            college = random.choice(self.fake_colleges)
            issue = "fake_college"
        else:
            college = random.choice(self.colleges)
            issue = fake_type
        
        name = random.choice(self.names)
        
        # Fake or malformed roll numbers
        if random.choice([True, False]):
            roll_no = f"{random.randint(1, 9)}{random.choice(['X', 'Y', 'Z'])}{random.randint(10, 99)}"  # Invalid format
        else:
            roll_no = str(random.randint(1000000, 9999999))  # Too long
        
        # Poor layout
        draw.text((width//2 - 60, 20), "STUDENT CARD", fill=(255, 0, 0), font=title_font)  # Wrong title
        
        y_offset = 60
        texts = [
            college,
            f"Student Name: {name}",
            f"ID Number: {roll_no}",
            f"Course: {random.choice(self.courses)}",
            "Valid: Forever",  # Unrealistic validity
            "Grade: A+++"  # Suspicious field
        ]
        
        for text in texts:
            draw.text((20, y_offset), text, fill=(0, 0, 0), font=font)
            y_offset += 20
        
        # Add obviously fake elements
        draw.text((width-100, height-30), "100% REAL", fill=(255, 0, 0), font=font)
        
        # Add some distortion to make it look manipulated
        if random.choice([True, False]):
            # Add pixelation effect
            small = image.resize((width//4, height//4), Image.NEAREST)
            image = small.resize((width, height), Image.NEAREST)
        
        metadata = {
            "category": "fake",
            "college": college,
            "name": name,
            "roll_no": roll_no,
            "fake_type": fake_type,
            "issues": [issue, "poor_design", "suspicious_fields"],
            "expected_label": "fake",
            "expected_status": "rejected"
        }
        
        return {"image": image, "metadata": metadata, "filename": f"fake_id_{index:03d}.jpg"}
    
    def generate_non_id_image(self, index: int) -> Dict[str, Any]:
        """Generate a non-ID image (meme, screenshot, etc.)"""
        width, height = random.choice([(500, 300), (400, 400), (600, 400)])
        
        # Random bright colors
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        bg_color = random.choice(colors)
        
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        font = self.get_default_font(20)
        
        # Random text that's clearly not an ID
        texts = [
            "MEME IMAGE",
            "This is not an ID card",
            "Just a random picture",
            "Screenshot of something",
            "🎉 Party Time! 🎉"
        ]
        
        text = random.choice(texts)
        
        # Center the text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        
        # Add some random shapes
        for _ in range(3):
            x1, y1 = random.randint(0, width//2), random.randint(0, height//2)
            x2, y2 = random.randint(width//2, width), random.randint(height//2, height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.ellipse([x1, y1, x2, y2], fill=color)
        
        metadata = {
            "category": "fake",
            "fake_type": "non_id",
            "issues": ["not_an_id", "inappropriate_content"],
            "expected_label": "fake",
            "expected_status": "rejected"
        }
        
        return {"image": image, "metadata": metadata, "filename": f"non_id_{index:03d}.jpg"}
    
    def save_image_and_metadata(self, data: Dict[str, Any]) -> str:
        """Save image and return base64 encoding"""
        category = data["metadata"]["category"]
        image_path = os.path.join(self.output_dir, category, data["filename"])
        
        # Save image
        data["image"].save(image_path, "JPEG", quality=85)
        
        # Convert to base64
        buffer = io.BytesIO()
        data["image"].save(buffer, format="JPEG")
        base64_string = base64.b64encode(buffer.getvalue()).decode()
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, "metadata", 
                                   data["filename"].replace(".jpg", ".json"))
        with open(metadata_file, "w") as f:
            json.dump(data["metadata"], f, indent=2)
        
        return base64_string
    
    def generate_dataset(self, num_each_category: int = 20) -> Dict[str, List[Dict]]:
        """Generate complete dataset"""
        dataset = {
            "genuine": [],
            "suspicious": [],
            "fake": []
        }
        
        print(f"Generating {num_each_category} samples for each category...")
        
        # Generate genuine IDs
        print("Generating genuine ID cards...")
        for i in range(num_each_category):
            data = self.generate_genuine_id(i)
            base64_string = self.save_image_and_metadata(data)
            
            dataset["genuine"].append({
                "filename": data["filename"],
                "base64": base64_string,
                "metadata": data["metadata"]
            })
        
        # Generate suspicious IDs
        print("Generating suspicious ID cards...")
        for i in range(num_each_category):
            data = self.generate_suspicious_id(i)
            base64_string = self.save_image_and_metadata(data)
            
            dataset["suspicious"].append({
                "filename": data["filename"],
                "base64": base64_string,
                "metadata": data["metadata"]
            })
        
        # Generate fake IDs
        print("Generating fake ID cards...")
        for i in range(num_each_category):
            data = self.generate_fake_id(i)
            base64_string = self.save_image_and_metadata(data)
            
            dataset["fake"].append({
                "filename": data["filename"],
                "base64": base64_string,
                "metadata": data["metadata"]
            })
        
        # Save complete dataset
        dataset_file = os.path.join(self.output_dir, "complete_dataset.json")
        with open(dataset_file, "w") as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\nDataset generated successfully!")
        print(f"Total images: {len(dataset['genuine']) + len(dataset['suspicious']) + len(dataset['fake'])}")
        print(f"Genuine: {len(dataset['genuine'])}")
        print(f"Suspicious: {len(dataset['suspicious'])}")
        print(f"Fake: {len(dataset['fake'])}")
        print(f"Dataset saved to: {self.output_dir}")
        
        return dataset

def main():
    """Generate the test dataset"""
    generator = IDCardDatasetGenerator()
    
    # Generate 50 samples of each category (150 total)
    dataset = generator.generate_dataset(num_each_category=50)
    
    print("\nDataset structure:")
    print(f"📁 {generator.output_dir}/")
    print("  📁 genuine/")
    print("  📁 suspicious/")
    print("  📁 fake/")
    print("  📁 metadata/")
    print("  📄 complete_dataset.json")
    
    return dataset

if __name__ == "__main__":
    dataset = main() 