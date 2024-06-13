# HumAnimate Benchmark pose-prompt pair data
HumAnimate benchmark evaluates text- and pose-guided video generative models.
It includes 1130 paired text prompts and pose sequences.
We provide keypoints to enable future research with different pose sequence styles.

#### Dataset Link
Pose sequence keypoints: <a href="https://github.com/humanimate/benchmark/blob/main/data/##"> Keypoints </a>
Pose-Prompt pairs:
<ul>
  <li>
    <a href="https://github.com/humanimate/benchmark/blob/main/data/TextPoseBench_codebook__openpose_keypose.csv"> For OpenPose keypose stick-figures </a> <br>
  </li>
  <li>
    <a href="https://github.com/humanimate/benchmark/blob/main/data/TextPoseBench_codebook__hrnet_stickfigure.csv"> For HRNet stick-figures </a>
  </li>
</ul>

#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:

(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **Anonymized for reviews, Team:** HumAnimate Authors
<!-- - **Dhruv Srivastava, Team:** Owner -->
<!-- - **Name, Team:** (Owner / Contributor / Manager) -->

## Authorship
### Publishers
#### Publishing Organization(s)
International Institute of Information Technology, Hyderabad (IIIT-H), India

#### Industry Type(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** industry types to which the publishing
organizations belong: -->
- Academic - Tech


#### Contact Detail(s)
<!-- scope: microscope -->
<!-- info: Provide publisher contact details: -->
- **Publishing POC:** Anonymized for reviews
- **Affiliation:** Anonymized for reviews
- **Contact:** Anonymized for reviews
- **Mailing List:** NA
- **Website:** https://github.com/humanimate/benchmark

### Dataset Owners
#### Team(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the groups or team(s) that own the dataset: -->

#### Contact Detail(s)
<!-- scope: periscope -->
<!-- info: Provide pathways to contact dataset owners: -->
- **Dataset Owner(s):** Anonymized for reviews
- **Affiliation:** Anonymized for reviews
- **Contact:** Anonymized for reviews
- **Group Email:** NA
- **Website:** https://github.com/humanimate/benchmark

#### Author(s)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:

(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- Anonymized for reviews
<!-- - Name, Title, Affiliation, YYYY -->
<!-- - Name, Title, Affiliation, YYYY -->
<!-- - Name, Title, Affiliation, YYYY -->

### Funding Sources
#### Institution(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
- Anonymized for reviews
<!-- - Name of Institution -->
<!-- - Name of Institution -->

#### Funding or Grant Summary(ies)
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
the creation, collection, or curation of the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- *For example, Institution 1 and institution 2 jointly funded this dataset as a
part of the XYZ data program, funded by XYZ grant awarded by institution 3 for
the years YYYY-YYYY.* -->
NA

<!-- **Additional Notes:** Add here -->

## Dataset Overview
#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Non-Sensitive Data about people
- Data about places and objects
- Synthetically generated data

#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->
Category | Data
--- | ---
Size of Dataset | 123456 MB <!-- re check this -->
Number of Instances | 1130
Number of Fields | 123456
Labeled Classes | NA
Number of Labels | NA
Average Labeles Per Instance | NA
Algorithmic Labels | NA
Human Labels | NA
Other Characteristics | NA

**Above:** Base data statistics for HumAnimate benchmark dataset.
<!-- Provide a caption for the above table of visualization. -->

<!-- **Additional Notes:** Add here. -->

#### Content Description
<!-- scope: microscope -->
<!-- info: Provide a short description of the content in a data point: -->

Includes a folder with pose sequences in OpenPose keypose and HRNet stickfigure style,
and a csv file with 1130 prompt and pose-sequence reference pairs with additional details such as
original_vid_fps, sampling_fps, pose_height_class, pose_height_value, translation,
pose_misc, pose_metadata_filename, person_placeholder, action_placeholder, \
place_placeholder and experiment_comment.

<!-- **Additional Notes:** Add here. -->

#### Descriptive Statistics
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each field.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Some statistics will be relevant for numeric data, for not for
strings. -->
Stats | Index | original_vid_fps | sampling_fps | pose_height_value
--- | --- | --- | --- | --- 
count |  1130.000000 |    1130.000000 |  1130.000000  |    1130.000000
mean  |  564.500000  |     24.761575  |   5.000000    |    351.667257
std   |  326.347208  |     5.655702   |   0.714273    |    40.292760
min   |    0.000000  |     15.000000  |   2.000000    |    124.000000
25%   |  282.250000  |     23.976000  |   5.000000    |    340.000000
50%   |  564.500000  |     25.000000  |   5.000000    |    351.000000
75%   |  846.750000  |     29.970000  |   5.000000    |    365.000000
max   | 1129.000000  |     29.976000  |   8.000000    |    505.000000

<!-- Statistic | Field Name | Field Name | Field Name | Field Name | Field Name | Field Name
--- | --- | --- | --- | --- | --- | ---
count |
mean |
std |
min |
25% |
50% |
75% |
max |
mode | -->

**Above:** Statistics for numerical columns in the provided pose-sequence and prompt pairs csv.

**Additional Notes:** Add here.?

### Sensitivity of Data
#### Sensitivity Type(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable*** data types present in the dataset: -->
<!-- - User Content -->
<!-- - User Metadata -->
<!-- - User Activity Data -->
<!-- - Identifiable Data -->
<!-- - S/PII -->
<!-- - Business Data -->
<!-- - Employee Data -->
<!-- - Pseudonymous Data -->
- Anonymous Data
<!-- - Health Data -->
<!-- - Children’s Data -->
<!-- - None -->
- Others (Please specify)
  - Pose sequences (human actions)
  - Pose keypoints

#### Field(s) with Sensitive Data
<!-- scope: periscope -->
<!-- info: List fields in the dataset that contain S/PII, and specify if their
collection was intentional or unintentional.

Use additional notes to capture any other relevant information or
considerations. -->
No senseitive data collected

<!-- **Intentional Collected Sensitive Data**

(S/PII were collected as a part of the
dataset creation process.)

Field Name | Description
--- | ---
Field Name | Type of S/PII
Field Name | Type of S/PII
Field Name | Type of S/PII

**Unintentionally Collected Sensitive Data**

(S/PII were not explicitly collected as a
part of the dataset creation process but
can be inferred using additional
methods.)

Field Name | Description
--- | ---
Field Name | Type of S/PII
Field Name | Type of S/PII
Field Name | Type of S/PII

**Additional Notes:** Add here -->

#### Security and Privacy Handling
<!-- scope: microscope -->
<!-- info: Summarize the measures or steps to handle sensitive data in this
dataset.

Use additional notes to capture any other relevant information or
considerations. -->

**Method**: The input videos were processed to extract the pose keypoints and painted into a pose control sequence for OpenPose keypose and HRNet stick figure. This disolves all the human identifiable information and protects their privacy.

<!-- **Method:** description

**Method:** description

**Method:** description

**Additional Notes:** Add here -->

#### Risk Type(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** risk types presenting from the
dataset: -->
<!-- - Direct Risk -->
<!-- - Indirect Risk -->
<!-- - Residual Risk -->
- No Known Risks

<!-- - Others (Please Specify) -->

#### Supplemental Link(s)
<!-- scope: periscope -->
<!-- info: Provide link(s) for documentation pertaining to sensitive data in
the dataset: -->
**Link Name or Document Type:** https://github.com/humanimate/benchmark

<!-- **Link Name or Document Type:** link -->

<!-- **Link Name or Document Type:** link -->

#### Risk(s) and Mitigation(s)
<!-- scope: microscope -->
<!-- info: Summarize the steps taken to identify and mitigate risks from PII
or sensitive information.

Use additional notes to capture any other relevant information or
considerations. -->
No identifiable risk with shared data.

<!-- Summarize here. Include links and metrics where applicable.

**Risk type:** Description + Mitigations

**Risk type:** Description + Mitigations

**Risk type:** Description + Mitigations

**Additional Notes:** Add here -->

### Dataset Version and Maintenance
#### Maintenance Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->
**Regularly Updated** - New versions of the dataset
will continue to be made available.

<!-- **Actively Maintained** - No new versions will be made
available, but this dataset will
be actively maintained,
including but not limited to
updates to the data. -->

<!-- **Limited Maintenance** - The data will not be updated,
but any technical issues will be
addressed.

**Deprecated** - This dataset is obsolete or is
no longer being maintained.
 -->
#### Version Details
<!-- scope: periscope -->
<!-- info: Provide details about **this** version of the dataset: -->
**Current Version:** 1.0

**Last Updated:** 06/2024

**Release Date:** 06/2024

#### Maintenance Plan
<!-- scope: microscope -->
<!-- info: Summarize the maintenance plan for the dataset:

Use additional notes to capture any other relevant information or
considerations. -->
We will utilize GitHub issues and pull requests to manage public queries and
feature requests. The authors will actively maintain the provided benchmark data and
its associated code.

**Versioning:** Current version (v1.0) of data includes pose control signals
in OpenPose, and HRNet input style formats. In future, may include other conditioning signals.

**Updates:** No immediate updates, may include pose control signals in different styles in future.

**Errors:** No errors.

<!-- **Feedback:** Summarize here. Include information about criteria for refreshing -->
or updating the dataset.

<!-- **Additional Notes:** Add here -->

#### Next Planned Update(s)
<!-- scope: periscope -->
<!-- info: Provide details about the next planned update: -->
No immediate plan for any update.
<!-- **Version affected:** 1.0

**Next data update:** MM/YYYY

**Next version:** 1.1

**Next version update:** MM/YYYY -->

<!-- #### Expected Change(s) -->
<!-- scope: microscope -->
<!-- info: Summarize the updates to the dataset and/or data that are expected
on the next update.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- **Updates to Data:** Summarize here. Include links, charts, and visualizations
as appropriate.

**Updates to Dataset:** Summarize here. Include links, charts, and
visualizations as appropriate.

**Additional Notes:** Add here -->

## Example of Data Points
#### Primary Data Modality
<!-- scope: telescope -->
<!-- info: Select **one**: -->
<!-- - Image Data -->
- Text Data
- Tabular Data
<!-- - Audio Data -->
- Video Data
<!-- - Time Series -->
<!-- - Graph Data -->
<!-- - Geospatial Data -->
<!-- - Multimodel (please specify) -->
<!-- - Unknown -->
<!-- - Others (please specify) -->

#### Sampling of Data Points
<!-- scope: periscope -->
<!-- info: Provide link(s) to data points or exploratory demos: -->
- Demo Link: https://humanimate.github.io/
<!-- - Typical Data Point Link -->
<!-- - Outlier Data Point Link -->
<!-- - Other Data Point Link -->
<!-- - Other Data Point Link -->

#### Data Fields
<!-- scope: microscope -->
<!-- info: List the fields in data points and their descriptions.

(Usage Note: Describe each field in a data point. Optionally use this to show
the example.) -->
 #  | Column   |               Non-Null Count |  Dtype  
--- | ------   |               -------------- |  -----  
 0  | Unnamed: 0            |   1130 non-null |  int64  
 1  | filename              |  1130 non-null  | object 
 2  | prompt                |  1130 non-null  | object 
 3  | original_vid_fps      |  1130 non-null  | float64
 4  | sampling_fps          |  1130 non-null  | int64  
 5  | pose_height_class     |  1130 non-null  | object 
 6  | pose_height_value     |  1130 non-null  | int64  
 7  | translation           |  1130 non-null  | object 
 8  | pose_misc             |  10 non-null    | object 
 9  | pose_metadata_filename |  1130 non-null |  object 
 10 | person_placeholder    |  1130 non-null  | object 
 11 | action_placeholder    |  1130 non-null  | object 
 12 | place_placeholder     |  1130 non-null  | object 
 13 | experiment_comment    |  1130 non-null  | object

**Above:** Field names and their types from provided Prompt and pose sequence reference csv.

<!-- **Additional Notes:** Add here -->

#### Typical Data Point
<!-- width: half -->
<!-- info: Provide an example of a typical data point and describe what makes
it typical.

**Use additional notes to capture any other relevant information or
considerations.** -->
Summarize here. Include any criteria for typicality of data point.

```
{
  "Unnamed: 0": "0",
  "filename": "001_StandUpAndSit_openpose_keypose_5_normal_L2Rst2_NA.npy",
  "prompt": "A man gets up and sits back on a chair in a park. The camera moves from left to right.",
  "original_vid_fps": "15",
  "sampling_fps": "5",
  "pose_height_class": "normal",
  "pose_height_value": "350",
  "translation": "L2Rst2",
  "pose_misc": "NA",
  "pose_metadata_filename": "001_StandUpAndSit",
  "person_placeholder": "A man",
  "action_placeholder": "gets up and sits back on a chair",
  "place_placeholder": "in a park",
  "experiment_comment": "camera_motion",
}
```

<!-- **Additional Notes:** Add here -->

<!-- #### Atypical Data Point -->
<!-- width: half -->
<!-- info: Provide an example of an outlier data point and describe what makes
it atypical.

**Use additional notes to capture any other relevant information or
considerations.** -->
<!-- Summarize here. Include any criteria for atypicality of data point.

```
{'q_id': '8houtx',
  'title': 'Why does water heated to room temperature feel colder than the air around it?',
  'selftext': '',
  'document': '',
  'subreddit': 'explainlikeimfive',
  'answers': {'a_id': ['dylcnfk', 'dylcj49'],
  'text': ["Water transfers heat more efficiently than air. When something feels cold it's because heat is being transferred from your skin to whatever you're touching. ... Get out of the water and have a breeze blow on you while you're wet, all of the water starts evaporating, pulling even more heat from you."],
  'score': [5, 2]},
  'title_urls': {'url': []},
  'selftext_urls': {'url': []},
  'answers_urls': {'url': []}}
```

**Additional Notes:** Add here -->

## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
<!-- - Monitoring -->
- Research
<!-- - Production -->
<!-- - Others (please specify) -->

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->
For example: `Computer Vision`, `Human Animation Generation`, `Pose guidance`,
`Text guidance`, `Benchmark`.


#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->
- Enhance the human animation generation technologies.
- Provide accessible systems to systematically evaluate text- and pose-guided video generation methods.

### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
<!-- - Safe for production use -->
- Safe for research use
<!-- - Conditional use - some unsafe applications -->
<!-- - Only approved use -->
<!-- - Others (please specify) -->

#### Suitable Use Case(s)
<!-- scope: periscope -->
<!-- info: Summarize known suitable and intended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should
look out for, or other relevant information or considerations. -->
**Suitable Use Case:** For evaluation of text- and pose-guided human animation generation methods.

<!-- **Suitable Use Case:** Summarize here. Include links where necessary.

**Suitable Use Case:** Summarize here. Include links where necessary.

**Additional Notes:** Add here -->

#### Unsuitable Use Case(s)
<!-- scope: microscope -->
<!-- info: Summarize known unsuitable and unintended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Unsuitable Use Case:** Alterations of prompt suite to generate unethical content.

<!-- **Unsuitable Use Case:** Summarize here. Include links where necessary.

**Unsuitable Use Case:** Summarize here. Include links where necessary.

**Additional Notes:** Add here -->

#### Research and Problem Space(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the specific problem space that this
dataset intends to address. -->
This dataset provides a structured mechanism to study and evaluate text- and pose- guided video generation methods.

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Guidelines & Steps:** Under reviw

<!-- **BiBTeX:**
```
@article{kuznetsova2020open,
  title={The open images dataset v4},
  author={Kuznetsova, Alina and Rom, Hassan and Alldrin, and others},
  journal={International Journal of Computer Vision},
  volume={128},
  number={7},
  pages={1956--1981},
  year={2020},
  publisher={Springer}
}
``` -->

<!-- **Additional Notes:** Add here -->

## Access, Rentention, & Wipeout
### Access
#### Access Type
<!-- scope: telescope -->
<!-- info: Select **one**: -->
<!-- - Internal - Unrestricted -->
<!-- - Internal - Restricted -->
- External - Open Access
<!-- - Others (please specify) -->

#### Documentation Link(s)
<!-- scope: periscope -->
<!-- info: Provide links that describe documentation to access this
dataset: -->
- Dataset Website URL: https://humanimate.github.io
- GitHub URL: https://github.com/humanimate/benchmark

#### Prerequisite(s)
<!-- scope: microscope -->
<!-- info: Please describe any required training or prerequisites to access
this dataset. -->
Training free. To be used only for Evaluation purposes.

#### Policy Link(s)
<!-- scope: periscope -->
<!-- info: Provide a link to the access policy: -->
<!-- - Direct download URL
- Other repository URL -->

Code to download data: https://github.com/humanimate/benchmark
```
...
```

#### Access Control List(s)
<!-- scope: microscope -->
<!-- info: List and summarize any access control lists associated with this
dataset. Include links where necessary.

Use additional notes to capture any other information relevant to accessing
the dataset. -->
**Access Control List:** Providing free public access under "GNU AFFERO GENERAL PUBLIC LICENSE, Version 3".

<!-- **Access Control List:** Write summary and notes here.

**Access Control List:** Write summary and notes here.

**Additional Notes:** Add here -->

### Retention
#### Duration
<!-- scope: periscope -->
<!-- info: Specify the duration for which this dataset can be retained: -->
Publicly available with no end date.

<!-- #### Policy Summary -->
<!-- scope: microscope -->
<!-- info: Summarize the retention policy for this dataset. -->
<!-- **Retention Plan ID:** Write here

**Summary:** Write summary and notes here -->

#### Process Guide
<!-- scope: periscope -->
<!-- info: Summarize any requirements and related steps to retain the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
For example: Simply download the provided zipped pose sequences and appropriate
csv file, and use the provided data loader to draw inferences from text- and pose-guided video generative methods.

<!-- This dataset compiles with [standard policy guidelines]. -->

<!-- **Additional Notes:** Add here -->

#### Exception(s) and Exemption(s)
<!-- scope: microscope -->
<!-- info: Summarize any exceptions and related steps to retain the dataset.
Include links where necessary.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- **Exemption Code:** `ANONYMOUS_DATA` /
`EMPLOYEE_DATA` / `PUBLIC_DATA` /
`INTERNAL_BUSINESS_DATA` /
`SIMULATED_TEST_DATA`
 -->
**Summary:** No exception or exemption.

÷**Additional Notes:** Add here

### Wipeout and Deletion
#### Duration
<!-- scope: periscope -->
<!-- info: Specify the duration after which this dataset should be deleted or
wiped out: -->
The data and code will not be wiped out.
We will use GitHub to host the data and associated code.

#### Deletion Event Summary
<!-- scope: microscope -->
<!-- info: Summarize the sequence of events and allowable processing for data
deletion.

Use additional notes to capture any other relevant information or
considerations. -->
**Sequence of deletion and processing events:**

- No intention to delete or take down the dataset and code.

<!-- **Additional Notes:** Add here -->

#### Acceptable Means of Deletion
<!-- scope: periscope -->
<!-- info: List the acceptable means of deletion: -->
- No intention to delete or take down the dataset and code.

<!-- #### Post-Deletion Obligations -->
<!-- scope: microscope -->
<!-- info: Summarize the sequence of obligations after a deletion event.

**Use additional notes to capture any other relevant information or
considerations.** -->
<!-- **Sequence of post-deletion obligations:** -->

<!-- - Summarize first obligation here
- Summarize second obligation here
- Summarize third obligation here

**Additional Notes:** Add here -->

<!-- #### Operational Requirement(s) -->
<!-- scope: periscope -->
<!-- info: List any wipeout integration operational requirements: -->
<!-- **Wipeout Integration Operational Requirements:** -->

<!-- - Write first requirement here
- Write second requirement here
- Write third requirement here -->

<!-- #### Exceptions and Exemptions -->
<!-- scope: microscope -->
<!-- info: Summarize any exceptions and related steps to a deletion event.

**Use additional notes to capture any other relevant information or
considerations.** -->
<!-- **Policy Exception bug:** [bug] -->

<!-- **Summary:** Write summary and notes here -->

<!-- **Additional Notes:** Add here -->

## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->
<!-- - API -->
- Artificially Generated
<!-- - Crowdsourced - Paid -->
<!-- - Crowdsourced - Volunteer -->
- Vendor Collection Efforts
- Scraped or Crawled
<!-- - Survey, forms, or polls -->
- Taken from other existing datasets
<!-- - Unknown -->
<!-- - To be determined -->
<!-- - Others (please specify) -->

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** Weizmann dataset, KTH-actions, YouTube-dance, UTD-MHAD,  Taichi, TikTok datasets and non-copyright YouTube videos.

<!-- **Platform:** [Platform Name], Describe platform here. Include links where relevant. -->

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [01 2024 - 05 2024]

**Primary modality of collection data:**

*Usage Note: Select one for this collection type.*

<!-- - Image Data -->
<!-- - Text Data -->
<!-- - Tabular Data -->
<!-- - Audio Data -->
- Video Data
<!-- - Time Series -->
<!-- - Graph Data -->
<!-- - Geospatial Data -->
<!-- - Unknown -->
<!-- - Multimodal (please specify) -->
<!-- - Others (please specify) -->

**Update Frequency for collected data:**

*Usage Note: Select one for this collection type.*

<!-- - Yearly
- Quarterly
- Monthly
- Biweekly
- Weekly
- Daily
- Hourly
- Static -->
- Others (please specify)
  - Collected only once.

<!-- **Additional Links for this collection:**

- [Access Policy]
- [Wipeout Policy]
- [Retention Policy] -->

<!-- **Additional Notes:** Add here -->

#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->
- **Source:** Weizmann dataset, https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html
- **Source:** KTH-Actions dataset, https://www.csc.kth.se/cvap/actions/
- **Source:** Youtube dance dataset https://github.com/NVlabs/few-shot-vid2vid
- **Source:** UTD-MHAD https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
- **Source:** Taichi https://github.com/AliaksandrSiarohin/first-order-model/blob/master/data/taichi-loading/README.md
- **Source:** TikTok https://www.yasamin.page/hdnet_tiktok
- **Source:** YouTube https://youtube.com

<!-- **Additional Notes:** Add here -->

#### Collection Cadence
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
**Static:** Data was collected once from multiple sources.

<!-- **Streamed:** Data is continuously acquired from single or multiple sources. -->

<!-- **Dynamic:** Data is updated regularly from single or multiple sources. -->

<!-- **Others:** Please specify -->

#### Data Integration
<!-- scope: periscope -->
<!-- info: List all fields collected from different sources, and specify if
they were included or excluded from the dataset.

Use additional notes to
capture any other relevant information or considerations.

(Usage Note: Duplicate and complete the following for each upstream
source.) -->
<!-- **Source** -->
Filename | Source
--- | ---
Stand-up and sit | UTD-MHAD
On-spot Jog      | UTD-MHAD
Standing wave    | Weizmann
JumpingJacks     | Youtube
Bend down        | UTD-MHAD
On-spot Dance    | TikTok dance
Squats           | Youtube
High Knees       | Youtube
Walk             | KTH
Rotation+Dance   | TikTok-dance
Cartwheel        | Youtube
Moonwalk         | Youtube-dance
Taichi           | Taichi-dataset
Knee-tuck Jumps  | Youtube
Lateral Jumps    | Youtube
Alt. Lunges      | Youtube

<!-- **Included Fields**

Data fields that were collected and are included in the dataset.

Field Name | Description
--- | ---
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.

**Additional Notes:** Add here

**Excluded Fields**

Data fields that were collected but are excluded from the dataset.

Field Name | Description
--- | ---
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.
Field Name | Describe here. Include links, data examples, metrics, visualizations where relevant.

**Additional Notes:** Add here -->

#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->
**Collection Method or Source**

**Description:** Manually downloaded and saved locally for pose keypoint extraction.

**Methods employed:** Manually downloaded.

**Tools or libraries:** Manually downloaded.

<!-- **Additional Notes:** Add here -->

### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- Selective hand picked actions based on their complexity and duration were selected from different datasets listed in previous sections.

<!-- - **Collection Method of Source:** Summarize data selection criteria here. Include links where available.
- **Collection Method of Source:** Summarize data selection criteria here. Include links where available.
- **Collection Method of Source:** Summarize data selection criteria here. Include links where available. -->

<!-- **Additional Notes:** Add here -->

<!-- #### Data Inclusion -->
<!-- scope: periscope -->
<!-- info: Summarize the data inclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- - **Collection Method of Source:** Summarize data inclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data inclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data inclusion criteria here. Include links where available. -->

<!-- **Additional Notes:** Add here -->

#### Data Exclusion
<!-- scope: microscope -->
<!-- info: Summarize the data exclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- Sampled were discarded if they were complex or not suitable for the benchmark.
<!-- - **Collection Method of Source:** Summarize data exclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data exclusion criteria here. Include links where available.
- **Collection Method of Source:** Summarize data exclusion criteria here. Include links where available.

**Additional Notes:** Add here -->

### Relationship to Source
#### Use & Utility(ies)
<!-- scope: telescope -->
<!-- info: Describe how the resulting dataset is aligned with the purposes,
motivations, or intended use of the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
Collected dataset utilizes very basic to few complex actions which are diverse enough to evaluate text- and pose-guided video generation methods.
<!-- - **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available. -->

<!-- **Additional Notes:** Add here -->

#### Benefit and Value(s)
<!-- scope: periscope -->
<!-- info: Summarize the benefits of the resulting dataset to its consumers,
compared to the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- The provided dataset along with the benchmarking system allows the users to systematically evaluate and assess the ability of text- and pose-guided video generation methods.
<!-- - **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available. -->

<!-- **Additional Notes:** Add here -->

#### Limitation(s) and Trade-Off(s)
<!-- scope: microscope -->
<!-- info: What are the limitations of the resulting dataset to its consumers,
compared to the upstream source(s)?

Break down by source type.<br><br>(Usage Note: Duplicate and complete the
following for each source type.) -->
- Limitation: Does not include pose sequences with subjects tilted/rotated at specific angles. This requires specialized hardware to capture the subjects at fixed angle consistently.
<!-- - **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available. -->

### Version and Maintenance
<!-- info: Fill this next row if this is not the first version of the dataset,
and there is no data card available for the first version -->
#### First Version
<!-- scope: periscope -->
<!-- info: Provide a **basic description of the first version** of this
dataset. -->
- **Release date:** 06/2024
- **Link to dataset:** HumAnimate benchmark dataset + v1.0
- **Status:** Actively Maintained]
- **Size of Dataset:** 100 MB
- **Number of Instances:** 1130

#### Note(s) and Caveat(s)
<!-- scope: microscope -->
<!-- info: Summarize the caveats or nuances of the first version of this
dataset that may affect the use of the current version.

Use additional notes to capture any other relevant information or
considerations. -->
Allows stress testing of models at very basic level.
<!-- Summarize here. Include links where available.

**Additional Notes:** Add here -->

#### Cadence
<!-- scope: telescope -->
<!-- info: Select **one**: -->
<!-- - Yearly -->
<!-- - Quarterly -->
- Monthly
<!-- - Biweekly -->
<!-- - Weekly -->
<!-- - Daily -->
<!-- - Hourly -->
<!-- - Static -->
<!-- - Others (please specify) -->

#### Last and Next Update(s)
<!-- scope: periscope -->
<!-- info: Please describe the update schedule: -->
- **Date of last update:** 13/06/2024
- **Total data points affected:** 1130
- **Data points updated:** 1130
- **Data points added:** 1130
- **Data points removed:** 0
- **Date of next update:** NA

#### Changes on Update(s)
<!-- scope: microscope -->
<!-- info: Summarize the changes that occur when the dataset is refreshed.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
Published the code and dataset for the first time.
<!-- - **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available.
- **Source Type:** Summarize here. Include links where available. -->

<!-- **Additional Notes:** Add here -->

## Human and Other Sensitive Attributes
#### Sensitive Human Attribute(s)
<!-- scope: telescope -->
<!-- info: Select **all attributes** that are represented (directly or
indirectly) in the dataset. -->
<!-- - Gender
- Socio-economic status
- Geography
- Language
- Age
- Culture
- Experience or Seniority
- Others (please specify) -->
No sensitive information provided/shared in HumAnimate benchmark.

#### Intentionality
<!-- scope: periscope -->
<!-- info: List fields in the dataset that contain human attributes, and
specify if their collection was intentional or unintentional.

Use additional notes to capture any other relevant information or
considerations. -->
**Intentionally Collected Attributes**

Human pose keypoints are extracted from the videos and the original videos are disregarded.

<!-- **Additional Notes:** Add here -->

**Unintentionally Collected Attributes**

Other than pose keypoints, no other information was extracted from the action videos.
<!-- Human attributes were not explicitly collected as a part of the dataset
creation process but can be inferred using additional methods. -->

<!-- Field Name | Description
--- | ---
Field Name | Human Attributed Collected
Field Name | Human Attributed Collected -->

<!-- **Additional Notes:** Add here -->

#### Rationale
<!-- scope: microscope -->
<!-- info: Describe the motivation, rationale, considerations or approaches
that caused this dataset to include the indicated human attributes.

Summarize why or how this might affect the use of the dataset. -->
This data and benchmark is created to gain meaningful insights in a structured way from text- and pose-guided video generation methods. This helps the community to improve the video generation models.

#### Source(s)
<!-- scope: periscope -->
<!-- info: List the sources of the human attributes.

Use additional notes to capture any other relevant information or
considerations. -->

- **Human Attribute:** nose keypoint location
- **Human Attribute:** right_eye keypoint location
- **Human Attribute:** left_eye keypoint location
- **Human Attribute:** right_ear keypoint location
- **Human Attribute:** left_ear keypoint location
- **Human Attribute:** right_shoulder keypoint location
- **Human Attribute:** left_shoulder keypoint location
- **Human Attribute:** right_elbow keypoint location
- **Human Attribute:** left_elbow keypoint location
- **Human Attribute:** right_wrist keypoint location
- **Human Attribute:** left_wrist keypoint location
- **Human Attribute:** right_hip keypoint location
- **Human Attribute:** left_hip keypoint location
- **Human Attribute:** right_knee keypoint location
- **Human Attribute:** left_knee keypoint location
- **Human Attribute:** right_ankle keypoint location
- **Human Attribute:** left_ankle keypoint location
- **Human Attribute:** torso keypoint location

<!-- **Additional Notes:** Add here -->

#### Methodology Detail(s)
<!-- scope: microscope -->
<!-- info: Describe the methods used to collect human attributes in the
dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->

**Human Attribute Method:** The video were passed to HRNet pose keypoint extractor to extract the pose sequences for each frame.

**Collection task:** Utilized Python scripts to extract the pose keypoints.

**Platforms, tools, or libraries:**

- Ubuntu22.04 OS with Python environment. Majorly built over PyTorch 
<!-- - [Platform, tools, or libraries]: Write description here
- [Platform, tools, or libraries]: Write description here -->

<!-- **Additional Notes:** Add here -->

#### Distribution(s)
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each human attribute,
noting key takeaways in the caption.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->
Extract 18 body keypoints-
```
0 : "nose",
1 : "right_eye",
2 : "left_eye",
3 : "right_ear",
4 : "left_ear",
5 : "right_shoulder",
6 : "left_shoulder",
7 : "right_elbow",
8 : "left_elbow",
9 : "right_wrist",
10 : "left_wrist",
11 : "right_hip",
12 : "left_hip",
13 : "right_knee",
14 : "left_knee",
15 : "right_ankle",
16 : "left_ankle",
17: "torso",
```
<!-- Human Attribute | Label or Class | Label or Class | Label or Class | Label or Class
--- | --- | --- | --- | ---
Count | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456
[Statistic] | 123456 | 123456 | 123456 | 123456 -->

<!-- **Above:** Provide a caption for the above table or visualization. -->
<!-- **Additional Notes:** Add here -->

#### Known Correlations
<!-- scope: periscope -->
<!-- info: Describe any known correlations with the indicated sensitive
attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate for each known correlation.) -->
[`field_name`, `field_name`]

**Description:**  No sensitive data collected or used.

**Impact on dataset use:** NA

<!-- **Additional Notes:** add here -->

#### Risk(s) and Mitigation(s)
<!-- scope: microscope -->
<!-- info: Summarize systemic or residual risks, performance expectations,
trade-offs and caveats because of human attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Duplicate and complete the following for each human attribute. -->
**Human Attribute**

Only keypoint body locations were used. No immediate risk to community by the dataset.

<!-- **Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations] -->

<!-- **Trade-offs, caveats, & other considerations:** Summarize here. Include
visualizations, metrics, or links where necessary. -->

<!-- **Additional Notes:** Add here -->

## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to use with other data
<!-- - Conditionally safe to use with other data
- Should not be used with other data
- Unknown
- Others (please specify) -->

#### Known Safe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: List the known datasets or data types and corresponding
transformations that **are safe to join or aggregate** this dataset with. -->
**Dataset or Data Type:** The provided dataset is very specific to HumAnimate benchmark setting. Hence cannot be directly merged with other datasets.

<!-- **Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary. -->

#### Best Practices
<!-- scope: microscope -->
<!-- info: Summarize best practices for using this dataset with other datasets
or data types.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- Summarize here. Include visualizations, metrics, demonstrative examples,
or links where necessary. -->
To be used with the provided dataloaders and only for evaluating text- and pose-guided video generation methods.

<!-- **Additional Notes:** Add here -->

#### Known Unsafe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with other
datasets" or "Should not be used with other datasets":

List the known datasets or data types and corresponding transformations that
are **unsafe to join or aggregate** with this dataset. -->
No unsafe data types.

<!-- **Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary.

**Dataset or Data Type:** Summarize here. Include visualizations, metrics,
or links where necessary. -->

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with
other datasets" or "Should not be used with
other datasets":

Summarize limitations of the dataset that introduce foreseeable risks when the
dataset is conjoined with other datasets.

Use additional notes to capture any other relevant information or
considerations. -->

**No rotation keypoints:** Does not include keypoint where subjects are tilted/rotated at certain angles. This requies controlled environment for capturing subjects at fixed angles while performing actions. 
<!-- 
**Limitation type:** Dataset or data type, description and recommendation.

**Limitation type:** Dataset or data type, description and recommendation.

**Additional Notes:** Add here -->

### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to form and/or sample
<!-- - Conditionally safe to fork and/or sample -->
<!-- - Should not be forked and/or sampled -->
<!-- - Unknown -->
<!-- - Others (please specify) -->

#### Acceptable Sampling Method(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** acceptable methods to sample this
dataset: -->
<!-- - Cluster Sampling
- Haphazard Sampling
- Multi-stage sampling
- Random Sampling
- Retrospective Sampling
- Stratified Sampling -->
- Systematic Sampling
<!-- - Weighted Sampling
- Unknown
- Unsampled
- Others (please specify) -->

#### Best Practice(s)
<!-- scope: microscope -->
<!-- info: Summarize the best practices for forking or sampling this dataset.

Use additional notes to capture any other relevant information or
considerations. -->
Use the provided dataloader to load the pose sequences and the relevant text prompt.

<!-- **Additional Notes:** Add here -->

#### Risk(s) and Mitigation(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sampled":

Summarize known or residual risks associated with forking and sampling methods
when applied to the dataset.

Use additional notes to capture any other
relevant information or considerations. -->
<!-- Summarize here. Include links and metrics where applicable. -->
No risk with sampling instances from given dataset.
<!-- **Risk Type:** [Description + Mitigations]

**Risk Type:** [Description + Mitigations]

**Risk Type:** [Description + Mitigations]

**Additional Notes:** Add here -->

<!-- #### Limitation(s) and Recommendation(s) -->
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sample":

Summarize the limitations that the dataset introduces when forking
or sampling the dataset and corresponding recommendations.

Use additional notes to capture any other relevant information or
considerations. -->
<!-- Summarize here. Include links and metrics where applicable.

**Limitation Type:** [Description + Recommendation]

**Limitation Type:** [Description + Recommendation]

**Limitation Type:** [Description + Recommendation]

**Additional Notes:** Add here
 -->
### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
<!-- - Training
- Testing -->
- Validation
<!-- - Development or Production Use
- Fine Tuning
- Others (please specify) -->

#### Notable Feature(s)
<!-- scope: periscope -->
<!-- info: Describe any notable feature distributions or relationships between
individual instances made explicit.

Include links to servers where readers can explore the data on their own. -->

**Exploration Demo:** https://humanimate.github.io/

<!-- **Notable Field Name:** Describe here. Include links, data examples, metrics,
visualizations where relevant.

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

#### Usage Guideline(s)
<!-- scope: microscope -->
<!-- info: Summarize usage guidelines or policies that consumers should be
aware of.

Use additional notes to capture any other relevant information or
considerations. -->
**Usage Guidelines:** For evaluating and improving the text- and pose-guided video generation methods.

**Approval Steps:** Released publicly, no additional approvals required.

**Reviewer:** Annonymized for review

<!-- **Additional Notes:** Add here -->

#### Distribution(s)
<!-- scope: periscope -->
<!-- info: Describe the recommended splits and corresponding criteria.

Use additional notes to capture any other
relevant information or considerations. -->

Set | Number of data points
--- | ---
Evaluation | 1130
<!-- Test | 62,563
Validation | 62,563
Dev | 62,563 -->

**Above:** Number of instances used for evaluation by HumAnimate benchmark.

<!-- **Additional Notes:** Add here -->

#### Known Correlation(s)
<!-- scope: microscope -->
<!-- info: Summarize any known correlations with
the indicated features in this dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate for each known
correlation.) -->
`field_name`, `field_name`

**Description:** The action specified in the pose sequences are mentioned in the relevant text prompt.

**Impact on dataset use:** Improve the ability of video models to generate realistic videos.

**Risks from correlation:** Generating realistic human animations could lead to misuse, such as creating deepfakes that violate personal privacy, or misrepresent individuals by placing them in fabricated scenarios. Sharing such videos online could lead to spread of manipulated content.

<!-- **Additional Notes:** Add here -->

#### Split Statistics
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide the sizes of each split. As appropriate, provide any
descriptive statistics for features. -->

Only for evaluation: 1130 pose-prompt pairs.
<!-- Statistic | Train | Test | Valid | Dev
--- | --- | --- | --- | ---
Count | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456
Descriptive Statistic | 123456 | 123456 | 123456 | 123456 -->

<!-- **Above:** Caption for table above. -->

## Transformations
<!-- info: Fill this section if any transformations were applied in the
creation of your dataset. -->
### Synopsis
#### Transformation(s) Applied
<!-- scope: telescope -->
<!-- info: Select **all applicable** transformations
that were applied to the dataset. -->
<!-- - Anomaly Detection -->
<!-- - Cleaning Mismatched Values -->
<!-- - Cleaning Missing Values -->
<!-- - Converting Data Types -->
<!-- - Data Aggregation -->
<!-- - Dimensionality Reduction -->
<!-- - Joining Input Sources -->
<!-- - Redaction or Anonymization -->
- Others (Please specify)
  - Pose keypoint extraction

#### Field(s) Transformed
<!-- scope: periscope -->
<!-- info: Provide the fields in the dataset that
were transformed.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied. Include the data types to
which fields were transformed.) -->

No transformations during extraction of keypoints.
<!-- **Transformation Type**

Field Name | Source & Target
--- | ---
Field Name | Source Field: Target Field
Field Name | Source Field: Target Field
... | ...

**Additional Notes:** Add here -->

#### Library(ies) and Method(s) Used
<!-- scope: microscope -->
<!-- info: Provide a description of the methods
used to transform or process the
dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied.) -->
<!-- **Transformation Type** -->

**Method:** Pose keypoints were extracted using HRNet pose detection model.

**Platforms, tools, or libraries:**
- Ubuntu22.04 OS with Python environment. Majorly built over PyTorch 
<!-- - Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

**Transformation Results:** No transformations during extraction of keypoints.

<!-- **Additional Notes:** Add here -->

### Breakdown of Transformations
<!-- info: Fill out relevant rows. -->

<!-- #### Cleaning Missing Value(s) -->
<!-- scope: telescope -->
<!-- info: Which fields in the data were missing
values? How many? -->
<!-- Summarize here. Include links where available.

**Field Name:** Count or description

**Field Name:** Count or description

**Field Name:** Count or description -->

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were missing values cleaned?
What other choices were considered? -->
**Method:** Pose keypoints were extracted using HRNet pose detection model.

**Platforms, tools, or libraries:**
- Ubuntu22.04 OS with Python environment. Majorly built over PyTorch 
<!-- - Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

<!-- #### Comparative Summary -->
<!-- scope: microscope -->
<!-- info: Why were missing values cleaned using
this method (over others)? Provide
comparative charts showing before
and after missing values were cleaned. -->
<!-- Summarize here. Include links, tables, visualizations where available. -->

<!-- **Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ... -->

<!-- **Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
No risks identified.

<!-- - **Risk Type:** Description + Mitigations
- **Risk Type:** Description + Mitigations
- **Risk Type:** Description + Mitigations -->

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
Purely for automated evaluation, no human in the loop.

<!-- #### Additional Considerations -->
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Cleaning Mismatched Value(s) -->
<!-- scope: telescope -->
<!-- info: Which fields in the data were corrected
for mismatched values? -->
<!-- Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description -->

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were incorrect or mismatched
values cleaned? What other choices
were considered? -->
**Method:** Pose keypoints were extracted using HRNet pose detection model.


<!-- #### Comparative Summary -->
<!-- scope: microscope -->
<!-- info: Why were incorrect or mismatched
values cleaned using this method (over
others)? Provide a comparative
analysis demonstrating before and
after values were cleaned. -->
<!-- Summarize here. Include links where available. -->

<!-- **Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Residual & Other Risk(s) -->
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
<!-- Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations -->

<!-- #### Human Oversight Measure(s) -->
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Additional Considerations -->
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Anomalies -->
<!-- scope: telescope -->
<!-- info: How many anomalies or outliers were
detected?
If at all, how were detected anomalies
or outliers handled?
Why or why not? -->
<!-- Summarize here. Include links where available. -->

<!-- **Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description -->

<!-- #### Method(s) Used -->
<!-- scope: periscope -->
<!-- info: What methods were used to detect
anomalies or outliers? -->
<!-- Summarize here. Include links where necessary. -->

<!-- **Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

<!-- #### Comparative Summary -->
<!-- scope: microscope -->
<!-- info: Provide a comparative analysis
demonstrating before and after
anomaly handling measures. -->
<!-- Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Residual & Other Risk(s) -->
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
<!-- Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations -->

<!-- #### Human Oversight Measure(s) -->
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Additional Considerations -->
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
<!-- Summarize here. Include links where available. -->

#### Dimensionality Reduction
<!-- scope: telescope -->
<!-- info: How many original features were
collected and how many dimensions
were reduced? -->
No dimensionality reduction.

<!-- **Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description
 -->
<!-- #### Method(s) Used -->
<!-- scope: periscope -->
<!-- info: What methods were used to reduce the
dimensionality of the data? What other
choices were considered? -->
<!-- Summarize here. Include links where
necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

<!-- #### Comparative Summary -->
<!-- scope: microscope -->
<!-- info: Why were features reduced using this
method (over others)? Provide
comparative charts showing before
and after dimensionality reduction
processes. -->
<!-- Summarize here. Include links, tables, visualizations where available. -->

<!-- **Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Residual & Other Risks -->
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
<!-- Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations -->

<!-- #### Human Oversight Measure(s) -->
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Additional Considerations -->
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Joining Input Sources -->
<!-- scope: telescope -->
<!-- info: What were the distinct input sources that were joined? -->
<!-- Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description -->

<!-- #### Method(s) Used -->
<!-- scope: periscope -->
<!-- info: What are the shared columns of fields used to join these
sources? -->
<!-- Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

<!-- #### Comparative Summary -->
<!-- scope: microscope -->
<!-- info: Why were features joined using this
method over others?

Provide comparative charts showing
before and after dimensionality
reduction processes. -->
<!-- Summarize here. Include links, tables, visualizations where available. -->

<!-- **Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Residual & Other Risk(s) -->
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
<!-- Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations -->

<!-- #### Human Oversight Measure(s) -->
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
<!-- Summarize here. Include links where
available.
 -->
<!-- #### Additional Considerations -->
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
<!-- Summarize here. Include links where
available.
 -->

#### Redaction or Anonymization
<!-- scope: telescope -->
<!-- info: Which features were redacted or
anonymized? -->
Pose keypoint extraction from the user videos.
<!-- 
**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description -->

<!-- #### Method(s) Used -->
<!-- scope: periscope -->
<!-- info: What methods were used to redact or
anonymize data? -->
<!-- Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

<!-- #### Comparative Summary -->
<!-- scope: microscope -->
<!-- info: Why was data redacted or anonymized
using this method over others? Provide
comparative charts showing before
and after redaction or anonymization
process. -->
<!-- Summarize here. Include links, tables, visualizations where available. -->

<!-- **Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here
 -->
<!-- #### Residual & Other Risk(s) -->
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
<!-- Summarize here. Include links and metrics where applicable.

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations

**Risk Type:** Description + Mitigations -->

<!-- #### Human Oversight Measure(s) -->
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Additional Considerations -->
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Others (Please Specify) -->
<!-- scope: telescope -->
<!-- info: What was done? Which features or
fields were affected? -->
<!-- Summarize here. Include links where available.

**Field Name:** Count or Description

**Field Name:** Count or Description

**Field Name:** Count or Description -->

<!-- #### Method(s) Used -->
<!-- scope: periscope -->
<!-- info: What method were used? -->
<!-- Summarize here. Include links where necessary.

**Platforms, tools, or libraries**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

<!-- #### Comparative Summary -->
<!-- scope: microscope -->
<!-- info: Why was this method used over
others? Provide comparative charts
showing before and after this
transformation. -->
<!-- Summarize here. Include links, tables, visualizations where available.

**Field Name** | **Diff**
--- | ---
Field Name | Before: After
Field Name | Before: After
... | ...

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Residual & Other Risk(s) -->
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
<!-- Summarize here. Include links and metrics where applicable.

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations]

**Risk type:** [Description + Mitigations] -->

<!-- #### Human Oversight Measure(s) -->
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
<!-- Summarize here. Include links where available. -->

<!-- #### Additional Considerations -->
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
<!-- Summarize here. Include links where available. -->

## Annotations & Labeling
<!-- info: Fill this section if any human or algorithmic annotation tasks were
performed in the creation of your dataset. -->
#### Annotation Workforce Type
<!-- scope: telescope -->
<!-- info: Select **all applicable** annotation
workforce types or methods used
to annotate the dataset: -->
<!-- - Annotation Target in Data
- Machine-Generated
- Annotations -->
- Human Annotations (Expert)
<!-- - Human Annotations (Non-Expert)
- Human Annotations (Employees)
- Human Annotations (Contractors)
- Human Annotations (Crowdsourcing)
- Human Annotations (Outsourced / Managed)
- Teams
- Unlabeled
- Others (Please specify) -->

#### Annotation Characteristic(s)
<!-- scope: periscope -->
<!-- info: Describe relevant characteristics of annotations
as indicated. For quality metrics, consider
including accuracy, consensus accuracy, IRR,
XRR at the appropriate granularity (e.g. across
dataset, by annotator, by annotation, etc.).

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Annotation Type** | **Number**
--- | ---
Number of unique annotations | 16
Total number of annotations | 16
Average annotations per example | 1
Number of annotators per example | 1

**Above:** Count of annotations- manually tagging the type of action performed in the 8 base and 8 advanced videos.

<!-- **Additional Notes:** Add here -->

#### Annotation Description(s)
<!-- scope: microscope -->
<!-- info: Provide descriptions of the annotations
applied to the dataset. Include links
and indicate platforms, tools or libraries
used wherever possible.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**Annotation Type:** Manually typing what action is performed in the given video.

**Description:** Given a video of a human performing some action, write the textual description of the action in English language.

**Link:** https://github.com/humanimate/benchmark/tree/main/data

**Platforms, tools, or libraries:**
- Ubuntu22.04 OS with Python environment. Majorly built over PyTorch 
<!-- - Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here -->

<!-- **Additional Notes:** Add here -->

#### Annotation Distribution(s)
<!-- scope: periscope -->
<!-- info: Provide a distribution of annotations for each
annotation or class of annotations using the
format below.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
Due to very small samples, the transcriptions were done by the authors themselves.
<!-- **Annotation Type** | **Number**
--- | ---
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)
Annotations (or Class) | 12345 (20%)

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

#### Annotation Task(s)
<!-- scope: microscope -->
<!-- info: Summarize each task type associated
with annotations in the dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each task type.) -->
**Task Type:** Transcription

**Task description:** Manually typing what action is performed in the given video.

<!-- **Task instructions:** Summarize here. Include links if available. -->

**Methods used:** VLC media player to watch the videos and `gedit` to type the actions.

**Inter-rater adjudication policy:** Only one annotator.

**Golden questions:** What action is performed in the given video?

<!-- **Additional notes:** Add here -->

### Human Annotators
<!-- info: Fill this section if human annotators were used. -->
#### Annotator Description(s)
<!-- scope: periscope -->
<!-- info: Provide a brief description for each annotator
pool performing the human annotation task.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
Authors themselves transcribed the actions performed in the selected videos.
<!-- **Annotation Type**

**Task type:** Summarize here. Include links if available.

**Number of unique annotators:** Summarize here. Include links if available.

**Expertise of annotators:** Summarize here. Include links if available.

**Description of annotators:** Summarize here. Include links if available.

**Language distribution of annotators:** Summarize here. Include links if
available.

**Geographic distribution of annotators:** Summarize here. Include links if
available.

**Summary of annotation instructions:** Summarize here. Include links if
available.

**Summary of gold questions:** Summarize here. Include links if available.

**Annotation platforms:** Summarize here. Include links if available.

**Additional Notes:** Add here -->

#### Annotator Task(s)
<!-- scope: microscope -->
<!-- info: Provide a brief description for each
annotator pool performing the human
annotation task.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**Task Type:** Transcription

**Task description:** Manually typing what action is performed in the given video.

<!-- **Task instructions:** Summarize here. Include links if available. -->

**Methods used:** VLC media player to watch the videos and `gedit` to type the actions.

**Inter-rater adjudication policy:** Only one annotator.

**Golden questions:** What action is performed in the given video?

#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and
complete the following for each
annotation type.) -->
**Annotation Type** Transcription

- English [100 %]
<!-- - Language [Percentage %]
- Language [Percentage %] -->

<!-- **Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

#### Location(s)
<!-- scope: periscope -->
<!-- info: Provide annotator distributions for each
annotation type.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**(Annotation Type)** Transcription

- Annonymized for review
<!-- - Location [Percentage %]
- Location [Percentage %] -->

<!-- **Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

#### Gender(s)
<!-- scope: microscope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->

Annonymized for review.
<!-- **(Annotation Type)**

- Gender [Percentage %]
- Gender [Percentage %]
- Gender [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

## Validation Types
<!-- info: Fill this section if the data in the dataset was validated during
or after the creation of your dataset. -->
#### Method(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
<!-- - Data Type Validation
- Range and Constraint Validation
- Code/cross-reference Validation
- Structured Validation
- Consistency Validation -->
- Not Validated
<!-- - Others (Please Specify) -->

<!-- #### Breakdown(s) -->
<!-- scope: periscope -->
<!-- info: Provide a description of the fields and data
points that were validated.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
<!-- **(Validation Type)**

**Number of Data Points Validated:** 12345

**Fields Validated**
Field | Count (if available)
--- | ---
Field | 123456
Field | 123456
Field | 123456

**Above:** Provide a caption for the above table or visualization. -->

<!-- #### Description(s) -->
<!-- scope: microscope -->
<!-- info: Provide a description of the methods used to
validate the dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
<!-- **(Validation Type)**

**Method:** Describe the validation method here. Include links where
necessary.

**Platforms, tools, or libraries:**

- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here
- Platform, tool, or library: Write description here

**Validation Results:** Provide results, outcomes, and actions taken because
of the validation. Include visualizations where available.

**Additional Notes:** Add here -->

<!-- ### Description of Human Validators -->
<!-- info: Fill this section if the dataset was validated using human
validators -->
<!-- #### Characteristic(s) -->
<!-- scope: periscope -->
<!-- info: Provide characteristics of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations. -->
<!-- **(Validation Type)**
- Unique validators: 12345
- Number of examples per validator: 123456
- Average cost/task/validator: $$$
- Training provided: Y/N
- Expertise required: Y/N -->

<!-- #### Description(s) -->
<!-- scope: microscope -->
<!-- info: Provide a brief description of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
<!-- **(Validation Type)**

**Validator description:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Validator selection criteria:** Summarize here. Include links if available.

**Training provided:** Summarize here. Include links if available.

**Additional Notes:** Add here
 -->
<!-- #### Language(s) -->
<!-- scope: telescope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
<!-- **(Validation Type)**

- Language [Percentage %]
- Language [Percentage %]
- Language [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Location(s) -->
<!-- scope: periscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
<!-- **(Validation Type)**

- Location [Percentage %]
- Location [Percentage %]
- Location [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Gender(s) -->
<!-- scope: microscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
<!-- **(Validation Type)**

- Gender [Percentage %]
- Gender [Percentage %]
- Gender [Percentage %]

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
<!-- - Cluster Sampling
- Haphazard Sampling
- Multi-stage Sampling
- Random Sampling
- Retrospective Sampling
- Stratified Sampling -->
- Systematic Sampling
<!-- - Weighted Sampling
- Unknown
- Unsampled
- Others (Please specify) -->

#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of each sampling
method used.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each sampling method
used.) -->
Evaluation set is tied to specific benchmark dimensions, therefore requires sequential sampling from a dedicated subset.
<!-- **(Sampling Type)** | **Number**
--- | ---
Upstream Source | Write here
Total data sampled | 123m
Sample size | 123
Threshold applied | 123k units at property
Sampling rate | 123
Sample mean | 123
Sample std. dev | 123
Sampling distribution | 123
Sampling variation | 123
Sample statistic | 123 -->

<!-- **Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

<!-- #### Sampling Criteria -->
<!-- scope: microscope -->
<!-- info: Describe the criteria used to sample data from
upstream sources.

Use additional notes to capture any other
relevant information or considerations. -->
<!-- - **Sampling method:** Summarize here. Include links where applicable.
- **Sampling method:** Summarize here. Include links where applicable.
- **Sampling method:** Summarize here. Include links where applicable. -->

## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->
First text- and pose-guided video generation benchmark.

#### ML Application(s)
<!-- scope: telescope -->
<!-- info: Provide a list of key ML tasks
that the dataset has been
used for.

Usage Note: Use comma-separated keywords. -->
To improve Human animation generation.

#### Evaluation Result(s)
<!-- scope: periscope -->
<!-- info: Provide the evaluation results from
models that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
Please go through the provided research paper transcript for results and discussions.
<!-- **(Model Name)**

**Model Card:** [Link to full model card]

Evaluation Results

- Accuracy: 123 (params)
- Precision: 123 (params)
- Recall: 123 (params)
- Performance metric: 123 (params)

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here -->

#### Evaluation Process(es)
<!-- scope: microscope -->
<!-- info: Provide a description of the evaluation process for
the model's overall performance or the
determination of how the dataset contributes to
the model's performance.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model and method used.) -->
Please go through the provided research paper transcript for results and discussions.

<!-- **(Model Name)**

**[Method used]:** Summarize here. Include links where available.

- **Process:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Factors:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Considerations:** Summarize here. Include links, diagrams, visualizations, tables as relevant.
- **Results:** Summarize here. Include links, diagrams, visualizations, tables as relevant.

**Additional Notes:** Add here -->

#### Description(s) and Statistic(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the model(s) and
task(s) that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
Please go through the provided research paper transcript for results and discussions.

<!-- **(Model Name)**

**Model Card:** Link to full model card

**Model Description:** Summarize here. Include links where applicable.

- Model Size: 123 (params)
- Model Weights: 123 (params)
- Model Layers 123 (params)
- Latency: 123 (params)

**Additional Notes:** Add here -->

#### Expected Performance and Known Caveats
<!-- scope: microscope -->
<!-- info: Provide a description of the expected performance
and known caveats of the models for this dataset.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model.) -->
Please go through the provided research paper transcript for results and discussions.

<!-- **(Model Name)**

**Expected Performance:** Summarize here. Include links where available.

**Known Caveats:** Summarize here. Include links, diagrams, visualizations, and
tables as relevant.

**Additioanl Notes:** Add here -->

## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
No special term was used in the HumAnimate benchmark dataset. It consists of 1130 paired text-prompts and pose sequences.
<!-- 
#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here
 -->
## Reflections on Data
<!-- info: Use this space to include any additional information about the
dataset that has not been captured by the Data Card. For example,
does the dataset contain data that might be offensive, insulting, threatening,
or might otherwise cause anxiety? If so, please contact the appropriate parties
to mitigate any risks. -->
- We provide 1130 paired text prompts and pose sequences.
- Pose sequences include OpenPose keypose and HRNet stickfigure style control sequences.
- We provide keypoints to enable future research with different pose sequence styles.
<!-- ### Title
Write notes here.

### Title
Write notes here.

### Title
Write notes here. -->
