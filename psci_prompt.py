# -*- coding: utf-8 -*-
"""
Xu et al. (2025) PSCI grading prompt — Module 1 of the DSS.

Copied verbatim from Model_5_Claude_Colab_New.ipynb so the deployed DSS uses
the identical prompt that was benchmarked against the Xu ground-truth datasets.
Do not edit these strings without re-running the benchmark: the reported MAE/MSE
only describe this exact prompt.

PSCI scale: 10 = newly constructed / no visible defects, 1 = completely failed.
"""

MODEL = "claude-sonnet-4-6"

road_grading_standards = """

    Road Condition Rating System (Pavement Surface Condition Index - PSCI):
    Rating 10
        Description: No visible defects. The road is newly constructed or recently overlaid and in perfect condition.
        Key Characteristics: No cracks, potholes, ravelling, or surface distortions. Surface is smooth and even.
        Expected Defects: None.
    Rating 9
        Description: Minor surface defects such as very small areas of ravelling or bleeding (less than 10% of the surface).
        Key Characteristics: The road may show minor loss of surface texture but is generally smooth and without significant structural damage.
        No structural distresses like cracking, rutting, or edge breakup.
        Expected Defects: Minor ravelling or slight bleeding, but very localized.
    Rating 8
        Description: Moderate surface defects such as ravelling or bleeding affecting 10-30% of the surface. No significant structural damage.
        Key Characteristics: Some surface wear is visible, but the overall road surface remains intact. No major cracks, rutting, or other structural defects.
        Expected Defects: Moderate ravelling or bleeding.
    Rating 7
        Description: Extensive surface defects where more than 30% of the surface shows ravelling or bleeding. The road is old but with little or no structural distress.
        Key Characteristics: Large areas of surface wear or bleeding.
        The road may look aged, but there are no major cracks or structural damage.
        Expected Defects: Extensive ravelling or bleeding over the surface.
    Rating 6
        Description: Moderate presence of other pavement defects such as cracking (less than 20% of the surface), surface distortion, or patching in good condition. No structural distresses like rutting or edge cracking.
        Key Characteristics: Visible surface distortions requiring reduced speeds. Cracking may be present but is not severe.
        Expected Defects: Minor cracks (longitudinal or transverse), some surface distortion, good quality patches.
    Rating 5
        Description: Significant other pavement defects affecting more than 20% of the surface. Some structural distresses like rutting, minor alligator cracking, or edge breakup may be visible, but they are localized.
        Key Characteristics: Frequent cracks, surface distortions, and patches. Minor localized structural damage.
        Expected Defects: Frequent cracking, some local rutting, small areas of alligator cracking or edge cracking.
    Rating 4
        Description: Structural distresses present over 5-25% of the surface, including rutting, alligator cracking, poor patching, or edge breakup. Frequent potholes are likely.
        Key Characteristics: The road requires structural overlay to strengthen it. Visible signs of rutting and cracking.
        Expected Defects: Alligator cracking, edge breakup, rutting, frequent potholes.
    Rating 3
        Description: Extensive structural distress covering 25-50% of the surface, with large areas of rutting, alligator cracking, or edge breakup. Potholes and failed patches are frequent.
        Key Characteristics: The road is significantly damaged and difficult to drive on. Large areas show signs of major defects.
        Expected Defects: Widespread alligator cracking, deep rutting, frequent and large potholes, edge disintegration.
    Rating 2
        Description: Over 50% of the road shows severe structural distresses, including rutting deeper than 75mm, large potholes, and very poor patching. The road is difficult to drive on and in need of full-depth reconstruction.
        Key Characteristics: The road is severely deteriorated. Major portions of the surface are cracked, distorted, or missing.
        Expected Defects: Severe rutting, extensive alligator cracking, large potholes, surface disintegration.
    Rating 1
        Description: The road has completely failed with extensive loss of surface material, large potholes, and patching in failed condition. It is undriveable and needs complete reconstruction.
        Key Characteristics: The road is virtually unusable with deep potholes, surface break-up, and complete structural failure.
        Expected Defects: Large potholes, road disintegration, complete surface failure.

    Here are the characteristics of the 10 road surface defects to help visually identify them in images:

    1. Ravelling (surface defects)
        Appearance: Surface looks rough with small aggregate particles missing; advanced ravelling shows a worn surface with larger aggregates exposed.
        Key Visual Clue: Loss of surface material, especially in wheel paths or edges.
    2. Bleeding (surface defects)
        Appearance: Shiny, reflective, and often sticky-looking patches on the road, especially in wheel paths.
        Key Visual Clue: Dark, shiny spots that appear wet or oily but are not due to moisture.
    3. Cracking (Longitudinal, Transverse, Reflection, Slippage) (pavement defects)
        Appearance: Longitudinal cracks run parallel to the road center; transverse cracks run perpendicular; slippage cracks are crescent-shaped.
        Key Visual Clue: Linear cracks either along the road or crossing it, with potential for further deterioration.
    4. Rutting (structural distresses)
        Appearance: Deep, long, narrow depressions in the wheel paths.
        Key Visual Clue: Visible grooves or channels where vehicles frequently pass.
    5. Alligator Cracking (structural distresses)
        Appearance: Cracks that form a grid or "chicken wire" pattern resembling alligator skin.
        Key Visual Clue: Small, interconnected cracks, often seen in areas with heavy traffic loading.
    6. Edge Cracking/Breakup (structural distresses)
        Appearance: Cracks that develop along the edges of the pavement, usually within 300mm from the edge.
        Key Visual Clue: Linear cracks or broken pavement material along the road edges.
    7. Surface Distortion (structural distresses)
        Appearance: Wavy or uneven road surface with depressions or bumps, but not necessarily in wheel paths.
        Key Visual Clue: Deformations that create dips, sags, or humps, reducing smoothness.
    8. Patching (structural distresses)
        Appearance: Sections of pavement that have been repaired with new material, often different in color or texture.
        Key Visual Clue: Visible patches with irregular borders; may have cracks or be sinking.
    9. Potholes (structural distresses)
        Appearance: Circular or irregular depressions where part of the pavement has crumbled away.
        Key Visual Clue: Bowl-shaped holes in the surface, sometimes with water or loose debris inside.
    10. Road Disintegration (structural distresses)
        Appearance: Large, rough, and cratered sections of the road where most of the pavement surface is gone.
        Key Visual Clue: Areas where the road surface is severely broken, exposing the underlying layers.

        """

system_prompt = (
    "You are a pavement engineer tasked with evaluating road conditions from images. "
    "You will also be provided with road condition grading criteria (road_grading_standards) and image. "
    "Use the following step-by-step instructions to evaluate the road grade. "
    "Step 1 - The user will provide you with road condition grading criteria. "
    "Step 2 - The user will provide you with an image, focus solely on the asphalt pavement. "
    "Step 3 - Please pay attention to any anomalies on the asphalt pavement, such as surface defects (ravelling, "
    "bleeding), pavement defects(Longitudinal cracks, Transverse crack), structural distresses(alligator crack, "
    "rutting, potholes, surface distortion, edge breakup, patching). "
    "When identifying road anomalies, exclude the effects of shadows and water stains caused by lighting conditions. "
    "Step 4 - Roughly estimate the proportion of each type of anomaly (surface defects, pavement defects and structural "
    "distresses) on the asphalt surface. This ratio will be used as the basis for subsequent road classification. "
    "Step 5 - Using the provided road condition grading criteria determine the grade of the road surface in the image. "
    "Step 6 - Hide your thought process and give me a numerical value. Respond with a single grade number based on the provided standards. "
    "Respond with only one number."
)


def build_user_content(base64_image, media_type):
    """The exact user-turn content block sequence used in the Model_5 benchmark."""
    return [
        {"type": "text", "text": "I will provide you with road condition grading criteria."},
        {"type": "text", "text": road_grading_standards},
        {"type": "text", "text": "I will provide you with an image, please focus solely on the asphalt pavement"},
        {"type": "image", "source": {"type": "base64",
                                     "media_type": media_type,
                                     "data": base64_image}},
        {"type": "text", "text": (
            "Please pay attention to any anomalies on the asphalt pavement, such as surface defects (ravelling, "
            "bleeding), pavement defects(Longitudinal cracks, Transverse crack), structural distresses(alligator crack, rutting, potholes, "
            "surface distortion, edge breakup, patching). "
            "When identifying road anomalies, exclude the effects of shadows and water stains caused by lighting conditions.")},
        {"type": "text", "text": (
            "Roughly estimate the proportion of each type of anomaly (surface defects, pavement defects and structural distresses) "
            "on the asphalt surface. This ratio will be used as the basis for subsequent road classification.")},
        {"type": "text", "text": "Using the provided road condition grading criteria determine the grade of the road surface in the image"},
        {"type": "text", "text": (
            "Hide your thought process and give me a numerical value. Respond with a single grade number based on the provided standards. "
            "Respond with only one number.")},
    ]
