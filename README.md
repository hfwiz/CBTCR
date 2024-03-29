# CBTCR
the code and dataset for CBTCR

#### MBFL-test-reduction-CBTCR-code-extend.py is the code for implementing CBTCR on Defects4J.

#### IJSEKE-CBTCR-analysis-extend.py is the process of experimental data for CBTCR and other techniques.

#### The intermediate data represents the output obtained from running Defects4J.

#### The subject programs used in the experiments are sourced from Defects4J。


We use the Defects4J from https://github.com/rjust/defects4j.

----------------------------------------------------------------
Defects4J is a collection of reproducible bugs and a supporting infrastructure
with the goal of advancing software engineering research.

Contents of Defects4J
================

The projects
---------------

[comment]: # (Do not edit; generated by framework/util/create_bugs_table.pl .)

Defects4J contains 835 bugs (plus 29 deprecated bugs) from the following open-source projects:

| Identifier      | Project name               | Number of active bugs | Active bug ids      | Deprecated bug ids (\*) |
|-----------------|----------------------------|----------------------:|---------------------|-------------------------| 
| Chart           | jfreechart                 |           26          | 1-26                | None                    |
| Cli             | commons-cli                |           39          | 1-5,7-40            | 6                       |
| Closure         | closure-compiler           |          174          | 1-62,64-92,94-176   | 63,93                   |
| Codec           | commons-codec              |           18          | 1-18                | None                    |
| Collections     | commons-collections        |            4          | 25-28               | 1-24                    |
| Compress        | commons-compress           |           47          | 1-47                | None                    |
| Csv             | commons-csv                |           16          | 1-16                | None                    |
| Gson            | gson                       |           18          | 1-18                | None                    |
| JacksonCore     | jackson-core               |           26          | 1-26                | None                    |
| JacksonDatabind | jackson-databind           |          112          | 1-112               | None                    |
| JacksonXml      | jackson-dataformat-xml     |            6          | 1-6                 | None                    |
| Jsoup           | jsoup                      |           93          | 1-93                | None                    |
| JxPath          | commons-jxpath             |           22          | 1-22                | None                    |
| Lang            | commons-lang               |           64          | 1,3-65              | 2                       |
| Math            | commons-math               |          106          | 1-106               | None                    |
| Mockito         | mockito                    |           38          | 1-38                | None                    |
| Time            | joda-time                  |           26          | 1-20,22-27          | 21                      |

\* Due to behavioral changes introduced under Java 8, some bugs are no longer
reproducible. Hence, Defects4J distinguishes between active and deprecated bugs:

- Active bugs can be accessed through `active-bugs.csv`.

- Deprecated bugs are removed from `active-bugs.csv`, but their metadata is
  retained in the project directory.

- Deprecated bugs can be accessed through `deprecated-bugs.csv`, which also
  details when and why a bug was deprecated.

We do not re-enumerate active bugs because publications using Defects4J artifacts
usually refer to bugs by their specific bug id.
